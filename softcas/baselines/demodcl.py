"""DeModCL homogeneous-detector port, primary Eq. 18--37 and Algorithm 1.

The module interface is an entire compatible detector plus its projection and
memory statistics. Hardware cost inputs are caller-bound estimates; this module
does not manufacture Orin measurements. No module eviction is introduced.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from .common import DetectorTask, evaluating, independent_models, snapshot, trainable


def normalize(value):
    return F.normalize(value,dim=-1,eps=1e-12)


class MemoryTrace:
    def __init__(self,dimension,capacity=500,new_prototype_distance=.25):
        if dimension<1 or capacity<1 or new_prototype_distance<0:
            raise ValueError('Positive trace dimensions/capacity and a nonnegative assignment radius required')
        self.dimension,self.capacity,self.threshold=dimension,capacity,new_prototype_distance
        self.prototypes=torch.empty(0,dimension)
        self.radii=torch.empty(0)
        self.counts=torch.empty(0,dtype=torch.long)
        self.n=0;self.mean=torch.zeros(dimension);self.m2=torch.zeros(dimension)

    def update(self,features):
        features=features.detach().float().cpu()
        if features.ndim!=2 or features.shape[1]!=self.dimension or not len(features) or not torch.isfinite(features).all():
            raise ValueError('Nonempty finite trace feature matrix required')
        old_count=self.n;batch_count=len(features)
        mean=features.mean(0);centered=((features-mean)**2).sum(0)
        delta=mean-self.mean
        self.mean=self.mean+delta*(batch_count/(old_count+batch_count))
        self.m2=self.m2+centered+delta.square()*(old_count*batch_count/(old_count+batch_count))
        self.n+=batch_count
        groups={}
        for value in normalize(features):
            if len(self.prototypes):
                distances=torch.linalg.vector_norm(self.prototypes-value,dim=1)
                nearest=int(distances.argmin());distance=float(distances[nearest])
            else:nearest,distance=0,math.inf
            if distance>self.threshold and len(self.prototypes)<self.capacity:
                self.prototypes=torch.cat([self.prototypes,value[None]])
                self.radii=torch.cat([self.radii,torch.zeros(1)])
                self.counts=torch.cat([self.counts,torch.ones(1,dtype=torch.long)])
            else:groups.setdefault(nearest,[]).append(value)
        for index,values in groups.items():
            batch=torch.stack(values);count=int(self.counts[index]);new=count+len(batch)
            prototype=normalize((count*self.prototypes[index]+batch.sum(0))/new)
            distance=torch.linalg.vector_norm(batch-prototype,dim=1).sum()
            # Streaming empirical-radius estimate over assigned updates. The
            # paper does not specify an assignment threshold; the explicit
            # detector-port threshold above is a DEV-configured parameter.
            self.radii[index]=(count*self.radii[index]+distance)/new
            self.prototypes[index]=prototype;self.counts[index]=new

    @property
    def variance(self):return self.m2/max(1,self.n)

    def distances(self,features):
        if not len(self.prototypes):return torch.full((len(features),),math.inf)
        return torch.cdist(normalize(features.detach().float().cpu()),self.prototypes).min(1).values

    def feasibility(self,features,*,beta_d,gamma1,coverage_tolerance,epsilon=1e-6):
        if not self.n:return 0.
        features=features.detach().float().cpu()
        mean,variance=features.mean(0),features.var(0,unbiased=False)
        distance=torch.linalg.vector_norm((mean-self.mean)/(self.variance.sqrt()+epsilon))
        distance=distance+beta_d*torch.linalg.vector_norm(torch.log(variance+epsilon)-torch.log(self.variance+epsilon))
        matrix=torch.cdist(normalize(features),self.prototypes)
        values,indices=matrix.min(1)
        # A zero-radius prototype must still cover its own floating-point
        # representation after normalization/serialization.
        coverage=(values<=coverage_tolerance*self.radii[indices]+epsilon).float().mean()
        return float(torch.exp(-gamma1*distance)*coverage)

    def state_dict(self):return {name:copy.deepcopy(getattr(self,name)) for name in ('prototypes','radii','counts','n','mean','m2')}

    def load_state_dict(self,state):
        for name,value in state.items():setattr(self,name,copy.deepcopy(value))
        if self.prototypes.shape[1]!=self.dimension or len(self.prototypes)>self.capacity:
            raise ValueError('Stored memory trace does not match the configured interface')


class AnchorMapping:
    def __init__(self,feature_dimension,anchor_dimension=128,histogram_bins=16,seed=0):
        self.features,self.dimension,self.bins=feature_dimension,anchor_dimension,histogram_bins
        generator=torch.Generator().manual_seed(seed)
        size=feature_dimension*2+histogram_bins
        self.projection=torch.randn(anchor_dimension,size,generator=generator)/math.sqrt(size)

    def __call__(self,features):
        features=features.detach().float().cpu()
        if features.ndim!=2 or features.shape[1]!=self.features or not len(features):raise ValueError('Invalid anchor feature matrix')
        histogram=torch.histc(normalize(features).flatten(),bins=self.bins,min=-1,max=1)
        histogram=histogram/histogram.sum().clamp_min(1)
        statistics=torch.cat([features.mean(0),features.std(0,unbiased=False),histogram])
        return normalize(self.projection@statistics)


def sparse_signature_contribution(anchor,sparsity,perturbation,generator):
    if not 0<sparsity<=1 or perturbation<0:raise ValueError('Valid signature sparsity/perturbation required')
    cutoff=torch.quantile(anchor.abs(),1-sparsity)
    dominant=(anchor.abs()>=cutoff).float()
    mask=normalize(dominant+perturbation*torch.randn(anchor.shape,generator=generator))
    return anchor*mask


def sharing_score(similarity,local_seconds,communication_seconds,finetune_seconds,balance,epsilon=1e-6):
    if not 0<=balance<=1 or min(local_seconds,communication_seconds,finetune_seconds)<0:
        raise ValueError('Nonnegative cost estimates and balance in [0,1] required')
    efficiency=min(1.,local_seconds/(communication_seconds+finetune_seconds+epsilon))
    return balance*max(0.,similarity)+(1-balance)*efficiency


class Module:
    def __init__(self,model,projection,trace,module_id,owner,anchor_dimension):
        self.model,self.projection,self.trace=model,projection,trace
        self.module_id,self.owner,self.version=module_id,owner,0
        self.signature_sum=torch.zeros(anchor_dimension)
        self.signature=torch.zeros(anchor_dimension)
        self.timestamp=0.

    def state_dict(self):
        return {'model':snapshot(self.model),'projection':snapshot(self.projection),'trace':self.trace.state_dict(),
                'module_id':self.module_id,'owner':self.owner,'version':self.version,
                'signature_sum':self.signature_sum,'signature':self.signature,'timestamp':self.timestamp}

    def packet(self):
        stream=io.BytesIO();torch.save(self.state_dict(),stream);return stream.getvalue()


class DeModCL:
    method='DeModCL'
    def __init__(self,models,*,learning_rate,local_steps,finetune_steps,costs,
                 maximum_task_seconds,storage_bytes,maximum_loss,seed,
                 module_threshold=.85,similarity_threshold=.70,sharing_threshold=.92,
                 gamma1=.4,gamma2=.5,anchor_weight=.5,anchor_dimension=128,
                 beta_d=1.,coverage_tolerance=1.,routing_beta=1.,signature_sparsity=.1,
                 signature_perturbation=.01,signature_ttl=3600.,prototype_capacity=500,
                 new_prototype_distance=.25,histogram_bins=16,initial_batches=1,
                 weight_decay=1e-4,task=None):
        independent_models(models)
        if len(costs)!=len(models) or min(maximum_task_seconds,storage_bytes,maximum_loss)<=0:
            raise ValueError('Bind positive per-node limits and a cost record for every node')
        for record in costs:
            if any(key not in record or not math.isfinite(record[key]) or record[key]<0 for key in ('local_seconds','finetune_seconds','bandwidth_bytes_per_second','metadata_seconds')):
                raise ValueError('Explicit finite local/finetune/bandwidth/control cost estimates required')
            if record['bandwidth_bytes_per_second']<=0:raise ValueError('Positive bandwidth required')
        self.models=models;self.prototypes=[copy.deepcopy(model) for model in models]
        self.task=task or DetectorTask()
        self.lr,self.local_steps,self.finetune_steps=learning_rate,local_steps,finetune_steps
        self.costs=copy.deepcopy(costs);self.max_time,self.capacity,self.max_loss=maximum_task_seconds,storage_bytes,maximum_loss
        self.threshold,self.matching,self.sharing=module_threshold,similarity_threshold,sharing_threshold
        self.gamma1,self.gamma2,self.anchor_weight=gamma1,gamma2,anchor_weight
        self.anchor_dimension,self.beta_d,self.coverage_tolerance=anchor_dimension,beta_d,coverage_tolerance
        self.routing_beta,self.rho,self.perturbation=routing_beta,signature_sparsity,signature_perturbation
        self.ttl,self.prototype_capacity,self.prototype_distance=signature_ttl,prototype_capacity,new_prototype_distance
        self.bins,self.initial_batches,self.weight_decay=histogram_bins,initial_batches,weight_decay
        self.seed=seed;self.generator=torch.Generator().manual_seed(seed)
        self.pools=[[] for _ in models];self.caches=[{} for _ in models]
        self.mapping=None;self.feature_dimension=None
        self.round_index=0;self.next_id=0

    def _features(self,model,data,initial=False):
        batches=data.require('D')
        chosen=batches[:self.initial_batches] if initial else batches
        with evaluating(model),torch.no_grad():
            return torch.cat([self.task.features(model,batch['img']).detach().float().cpu() for batch in chosen])

    def _new(self,owner):
        model=copy.deepcopy(self.prototypes[owner]);device=next(model.parameters()).device
        projection=nn.Linear(self.feature_dimension,self.anchor_dimension).to(device)
        trace=MemoryTrace(self.feature_dimension,self.prototype_capacity,self.prototype_distance)
        return Module(model,projection,trace,str(owner)+'_'+str(self.next_id),owner,self.anchor_dimension)

    def _restore(self,saved,owner=None):
        source=saved['owner'] if owner is None else owner
        module=self._new(source)
        module.model.load_state_dict(saved['model']);module.projection.load_state_dict(saved['projection'])
        module.trace.load_state_dict(saved['trace'])
        for name in ('module_id','owner','version','signature_sum','signature','timestamp'):
            setattr(module,name,copy.deepcopy(saved[name]))
        return module

    def _adapt(self,module,data,query,steps):
        module.model.train();trainable(module.model);module.projection.train()
        parameters=[p for p in module.model.parameters() if p.requires_grad]+list(module.projection.parameters())
        optimizer=torch.optim.Adam(parameters,lr=self.lr,weight_decay=self.weight_decay)
        query=query.to(next(module.model.parameters()).device)
        losses=[];batches=data.require('D')
        for step in range(steps):
            batch=batches[step%len(batches)];optimizer.zero_grad(set_to_none=True)
            task_loss,features=self.task.loss_features(module.model,batch)
            alignment=(module.projection(features)-query).square().sum(1).mean()
            loss=task_loss+self.anchor_weight*alignment
            if not torch.isfinite(loss):raise ValueError('Nonfinite DeModCL task/anchor objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('Nonfinite DeModCL gradient')
            optimizer.step();losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        with evaluating(module.model),torch.no_grad():
            final_loss=sum(float(self.task.loss_features(module.model,batch)[0]) for batch in batches)/len(batches)
        return final_loss,losses

    def _publish(self,owner,module,available,now):
        payload=module.packet();checksum=hashlib.sha256(payload).hexdigest()
        metadata={'owner':owner,'module_id':module.module_id,'version':module.version,'timestamp':now,
                  'bytes':len(payload),'signature':module.signature.clone(),'sha256':checksum,
                  'interface':[(name,tuple(value.shape)) for name,value in module.model.state_dict().items()]}
        for peer in range(len(self.models)):
            if peer!=owner and (available is None or available[peer][owner]):
                self.caches[peer][(owner,module.module_id)]=copy.deepcopy(metadata)
        return len(payload)

    def round(self,fit_data,select_data=None,available=None,now=None):
        now=time.time() if now is None else float(now)
        reports=[]
        # A slot is accounting only. Each receiver sees signatures published
        # before its own decision, including earlier arrivals in the same slot.
        for owner,data in enumerate(fit_data):
            initial=self._features(self.models[owner],data,True)
            if self.mapping is None:
                self.feature_dimension=initial.shape[1]
                self.mapping=AnchorMapping(self.feature_dimension,self.anchor_dimension,self.bins,self.seed)
            if initial.shape[1]!=self.feature_dimension:raise ValueError('Direct DeModCL module transfer requires compatible feature interfaces')
            query=self.mapping(initial);pool=self.pools[owner];cost=self.costs[owner]
            occupied=sum(len(module.packet()) for module in pool)
            scores=[module.trace.feasibility(initial,beta_d=self.beta_d,gamma1=self.gamma1,coverage_tolerance=self.coverage_tolerance) for module in pool]
            index=max(range(len(scores)),key=lambda i:(scores[i],-i)) if scores else None
            decision=None;module=None;communication=0;selection_score=None
            if index is not None and scores[index]>=self.threshold and cost['finetune_seconds']+cost['metadata_seconds']<=self.max_time and occupied<=self.capacity:
                module=self._restore(pool[index].state_dict());decision='reuse';steps=self.finetune_steps
            else:
                # Stale metadata is discarded. Relevance is metadata-only; no
                # remote raw samples or remote task labels enter the query.
                cache=self.caches[owner]
                for key in list(cache):
                    if now-cache[key]['timestamp']>self.ttl:del cache[key]
                candidates=[]
                interface=[(name,tuple(value.shape)) for name,value in self.prototypes[owner].state_dict().items()]
                for key,entry in cache.items():
                    source,_=key
                    if available is not None and not available[owner][source]:continue
                    if entry['interface']!=interface:continue
                    similarity=float(F.cosine_similarity(query[None],entry['signature'][None]))
                    if similarity<self.matching:continue
                    comm=entry['bytes']/cost['bandwidth_bytes_per_second']+cost['metadata_seconds']
                    if comm+cost['finetune_seconds']>self.max_time or occupied+entry['bytes']>self.capacity:continue
                    score=sharing_score(similarity,cost['local_seconds'],comm,cost['finetune_seconds'],self.gamma2)
                    candidates.append((score,key,entry))
                if candidates:
                    best,key,entry=max(candidates,key=lambda row:(row[0],-row[1][0],row[1][1]))
                    if best>=self.sharing:
                        source=next((m for m in self.pools[key[0]] if m.module_id==key[1]),None)
                        if source is not None:
                            payload=source.packet()
                            if hashlib.sha256(payload).hexdigest()==entry['sha256']:
                                saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
                                module=self._restore(saved,owner);module.owner=owner
                                module.module_id=str(owner)+'_'+str(self.next_id)
                                decision='share';steps=self.finetune_steps;communication=len(payload);selection_score=best
                if module is None:
                    candidate=self._new(owner)
                    if cost['local_seconds']+cost['metadata_seconds']<=self.max_time and occupied+len(candidate.packet())<=self.capacity:
                        module=candidate;decision='new';steps=self.local_steps
            if module is None:
                reports.append({'client':owner,'status':'infeasible_C1_C2','decision':None});continue
            final_loss,losses=self._adapt(module,data,query,steps)
            if not math.isfinite(final_loss) or final_loss>self.max_loss:
                reports.append({'client':owner,'status':'infeasible_C3','decision':decision,'loss':final_loss,
                                'signature_published':False,'received_module_bytes':communication});continue
            post_features=self._features(module.model,data)
            post=self.mapping(post_features)
            module.trace.update(post_features)
            module.signature_sum=module.signature_sum+sparse_signature_contribution(post,self.rho,self.perturbation,self.generator)
            module.signature=normalize(module.signature_sum);module.timestamp=now;module.version+=1
            # Trace growth is charged to storage, too. No hidden eviction is used.
            revised=occupied-(len(pool[index].packet()) if decision=='reuse' else 0)+len(module.packet())
            if revised>self.capacity:
                reports.append({'client':owner,'status':'infeasible_C2_after_trace_growth','decision':decision,'signature_published':False});continue
            if decision=='reuse':pool[index]=module
            else:pool.append(module);self.next_id+=1
            self.models[owner]=module.model
            published=self._publish(owner,module,available,now)
            reports.append({'client':owner,'status':'valid','decision':decision,'loss':final_loss,'losses':losses,
                            'feasibility_scores':scores,'sharing_score':selection_score,'module_bytes':published,
                            'received_module_bytes':communication,'signature_published':True,'module_count':len(pool)})
        self.round_index+=1
        return {'round':self.round_index,'clients':reports,'selection_E':0,'time_unit':'asynchronous task-arrival slot; not a federated synchronization round'}

    def predict(self,client,images):
        pool=self.pools[client]
        if not pool:
            dense=self.task.dense(self.models[client],images)
            return self.task.predictions_from_dense(self.models[client],dense)
        with torch.no_grad():
            features=self.task.features(self.models[client],images)
            distances=torch.stack([module.trace.distances(features) for module in pool],dim=1)
            weights=(-self.routing_beta*distances).softmax(1).to(images.device)
            values=[self.task.dense(module.model,images) for module in pool]
            if any(value.shape!=values[0].shape for value in values):raise ValueError('Compatible native dense grids required for soft module routing')
            mixed=sum(value*weights[:,i,None,None] for i,value in enumerate(values))
            return self.task.predictions_from_dense(pool[0].model,mixed)

    def state_dict(self):
        return {'method':self.method,'models':[snapshot(model) for model in self.models],
                'pools':[[module.state_dict() for module in pool] for pool in self.pools],
                'caches':copy.deepcopy(self.caches),'generator':self.generator.get_state(),
                'projection':None if self.mapping is None else self.mapping.projection,
                'feature_dimension':self.feature_dimension,'round_index':self.round_index,'next_id':self.next_id}

    def load_state_dict(self,state):
        if state['method']!=self.method or len(state['pools'])!=len(self.pools):raise ValueError('Wrong modular algorithm checkpoint')
        self.feature_dimension=state['feature_dimension']
        if self.feature_dimension is not None:
            self.mapping=AnchorMapping(self.feature_dimension,self.anchor_dimension,self.bins,self.seed)
            self.mapping.projection=state['projection'].clone()
            self.pools=[[self._restore(saved,owner) for saved in pool] for owner,pool in enumerate(state['pools'])]
        for model,saved in zip(self.models,state['models']):model.load_state_dict(saved)
        self.caches=copy.deepcopy(state['caches']);self.generator.set_state(state['generator'])
        self.round_index,self.next_id=state['round_index'],state['next_id']
