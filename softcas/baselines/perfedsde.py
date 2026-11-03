"""PerFedSDE-NAS search, archive training and nonzero-coordinate aggregation."""
from __future__ import annotations
import copy
import math
import random
from concurrent.futures import ThreadPoolExecutor

import torch

from .common import DetectorTask, independent_models, native_loss, restore_models, snapshot, trainable
from .nas_space import DetectorSupernet


class RBF:
    """Gaussian RBF interpolation of the truly evaluated local population."""
    def __init__(self,architectures,scores,smoothing=1e-6):
        self.x=torch.tensor(architectures,dtype=torch.float64)
        y=torch.tensor(scores,dtype=torch.float64)
        if len(self.x)!=len(y) or not len(y) or not torch.isfinite(y).all():
            raise ValueError('Nonempty finite architecture/score population required')
        distance=torch.cdist(self.x,self.x)
        positive=distance[distance>0]
        self.scale=float(positive.median()) if positive.numel() else 1.
        kernel=torch.exp(-(distance/self.scale)**2)
        self.weights=torch.linalg.solve(kernel+smoothing*torch.eye(len(y),dtype=y.dtype),y)

    def predict(self,architectures):
        values=torch.tensor(architectures,dtype=torch.float64)
        return torch.exp(-(torch.cdist(values,self.x)/self.scale)**2)@self.weights


def nonzero_aggregate(original,clients,sample_weights):
    if not clients or len(clients)!=len(sample_weights) or any(not math.isfinite(w) or w<=0 for w in sample_weights):
        raise ValueError('Nonempty supernet updates and positive sample weights required')
    result={name:value.clone() for name,value in original.items()}
    for state in clients:
        if state.keys()!=original.keys() or any(state[name].shape!=original[name].shape for name in original):
            raise ValueError('Supernet state names/shapes differ')
    for name,base in original.items():
        if not base.is_floating_point():
            continue
        numerator,denominator=torch.zeros_like(base),torch.zeros_like(base)
        for weight,state in zip(sample_weights,clients):
            difference=state[name].to(base.device)-base
            if not torch.isfinite(difference).all():raise ValueError('Nonfinite supernet update')
            numerator.add_(difference,alpha=weight)
            denominator.add_((difference!=0).to(base.dtype),alpha=weight)
        result[name]=base+torch.where(denominator>0,numerator/denominator.clamp_min(torch.finfo(base.dtype).tiny),0)
    return result


class PerFedSDENAS:
    method='PerFedSDE-NAS'
    def __init__(self,models,*,port_indices,generator,dataset_descriptors,learning_rate,
                 search_iterations,generated_candidates,offspring,archive_size,seed,
                 sample_weights=None,score_fn=None,parallel_generation=False,image_size=640):
        independent_models(models)
        if not generator.prepared:
            raise ValueError('PerFedSDE-NAS requires actually prepared and bound score/predictor weights')
        if min(search_iterations,generated_candidates,offspring,archive_size)<1:
            raise ValueError('Positive search, candidate, offspring and archive sizes required')
        self.models=models
        self.models[:]=[DetectorSupernet(model,port_indices,image_size) for model in models]
        self.global_state=snapshot(self.models[0])
        self.population=[{} for _ in models]
        self.descriptors=dataset_descriptors
        if len(dataset_descriptors)!=len(models):raise ValueError('One D-only descriptor per client required')
        self.generators=[copy.deepcopy(generator) for _ in models]
        for i,item in enumerate(self.generators):item.generator.manual_seed(seed+i)
        self.rng=random.Random(seed)
        self.lr,self.searches,self.generated=learning_rate,search_iterations,generated_candidates
        self.offspring,self.archive=offspring,archive_size
        self.weights=[1.]*len(models) if sample_weights is None else list(sample_weights)
        self.score_fn=score_fn or DetectorTask().ap
        self.parallel=parallel_generation
        self.round_index=0
        self.stage=0
        self.descriptors_by_stage={}

    def begin_stage(self,fit_data,stage=None,dataset_descriptors=None):
        for data in fit_data:data.require('D')
        stage=self.stage+1 if stage is None else stage
        descriptors=dataset_descriptors
        if descriptors is None:descriptors=self.descriptors_by_stage.get(str(stage))
        if descriptors is None and self.stage==0:descriptors=self.descriptors
        if descriptors is None or len(descriptors)!=len(self.models):
            raise ValueError('Bind current-stage D-only NAS descriptors; old-domain descriptors cannot be reused silently')
        self.descriptors=descriptors
        # A domain change invalidates old-domain fitness values. Supernet weights
        # are inherited; the new V population is evaluated afresh. Within this
        # stage the paper's inherited-population search remains unchanged.
        self.population=[{} for _ in self.models]
        self.stage=stage
        return {'stage':stage,'old_domain_fitness_cleared':True,'descriptor_role':'D'}

    def _initial_population(self,model,selection,population):
        evaluations=0
        count=3*model.dimensions
        while len(population)<count:
            architecture=tuple(self.rng.randrange(2) for _ in range(model.dimensions))
            if not any(architecture[:6]) or architecture in population:continue
            model.set_architecture(architecture)
            population[architecture]=float(self.score_fn(model,selection))
            evaluations+=1
        return evaluations

    def _epoch(self,model,data,archive):
        batches=data.require('D')
        model.train();trainable(model)
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=self.lr)
        losses=[]
        for batch in batches:
            model.set_architecture(self.rng.choice(archive))
            optimizer.zero_grad(set_to_none=True)
            loss=native_loss(model,batch)
            if not torch.isfinite(loss):raise ValueError('Nonfinite archive-supernet objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite archive-supernet gradient')
            optimizer.step();losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        return losses

    def round(self,fit_data,select_data,available=None):
        if len(fit_data)!=len(self.models) or len(select_data)!=len(self.models):
            raise ValueError('One D/V stream per search client required')
        # This reference retains the paper's coordinating server. Contact
        # restrictions belong to the study adapter; they cannot silently change
        # a server aggregation baseline into a decentralized CIRA search.
        states,reports=[],[]
        for i,(model,data,selection) in enumerate(zip(self.models,fit_data,select_data)):
            data.require('D');selection.require('V')
            model.load_state_dict(self.global_state)
            population=self.population[i]
            evaluations=self._initial_population(model,selection,population)
            searches=[]
            for _ in range(self.searches):
                ordered=sorted(population,key=lambda key:(-population[key],key))
                predictor=RBF(ordered,[population[key] for key in ordered])
                archive=ordered[:self.archive]
                if self.parallel:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pending=pool.submit(self.generators[i].generate,self.descriptors[i],self.generated)
                        losses=self._epoch(model,data,archive)
                        generated=pending.result()
                else:
                    generated=self.generators[i].generate(self.descriptors[i],self.generated)
                    losses=self._epoch(model,data,archive)
                candidates=list(dict.fromkeys(architecture for architecture in generated if architecture not in population))
                if candidates:
                    predicted=predictor.predict(candidates)
                    ranked=sorted(range(len(candidates)),key=lambda j:(-float(predicted[j]),candidates[j]))[:self.offspring]
                    for index in ranked:
                        architecture=candidates[index]
                        model.set_architecture(architecture)
                        score=float(self.score_fn(model,selection))
                        if not math.isfinite(score):raise ValueError('Nonfinite architecture selection score')
                        population[architecture]=score;evaluations+=1
                ordered=sorted(population,key=lambda key:(-population[key],key))[:3*model.dimensions]
                population={architecture:population[architecture] for architecture in ordered}
                searches.append({'archive':archive,'fitting_batches':len(losses),'losses':losses,
                                 'generated':len(generated),'new_unique':len(candidates)})
            self.population[i]=population
            best=min(population,key=lambda key:(-population[key],key))
            model.set_architecture(best)
            states.append(snapshot(model))
            reports.append({'client':i,'best_architecture':best,'best_selection_score':population[best],
                            'selection_E':evaluations,'searches':searches})
        self.global_state=nonzero_aggregate(self.global_state,states,self.weights)
        self.round_index+=1
        return {'round':self.round_index,'clients':reports,'selection_E':sum(row['selection_E'] for row in reports),
                'topology':'server_aggregation_reference','generation_parallel_to_training':self.parallel}

    def state_dict(self):
        return {'method':self.method,'models':[snapshot(m) for m in self.models],'global_state':self.global_state,
                'architectures':[m.architecture for m in self.models],'population':copy.deepcopy(self.population),
                'rng':self.rng.getstate(),'generators':[item.state_dict() for item in self.generators],
                'round_index':self.round_index,'stage':self.stage,'descriptors':self.descriptors}

    def load_state_dict(self,state):
        if state['method']!=self.method:raise ValueError('Wrong search algorithm checkpoint')
        restore_models(self.models,state['models'])
        for model,architecture in zip(self.models,state['architectures']):model.set_architecture(tuple(architecture))
        self.global_state,self.population=state['global_state'],copy.deepcopy(state['population'])
        self.rng.setstate(state['rng'])
        for item,saved in zip(self.generators,state['generators']):item.load_state_dict(saved)
        self.round_index=state['round_index']
        self.stage,self.descriptors=state['stage'],state['descriptors']
