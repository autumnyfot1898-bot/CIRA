"""Trainable operation-graph score/meta models and VE guided architecture SDE.

The detector port supplies an explicit seven-node DAG, computation-matched pair
pretraining and categorical operation reconstruction. Detector-specific encoder
weights must be prepared; no classification-search-space weights are invented.
"""
from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn import functional as F


class GraphScore(nn.Module):
    def __init__(self, dimensions, categories, hidden=64, layers=2, heads=4):
        super().__init__()
        self.dimensions, self.categories, self.hidden = dimensions, categories, hidden
        self.operation = nn.Linear(categories, hidden)
        self.position = nn.Parameter(torch.randn(dimensions, hidden) * .02)
        self.time = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=0., batch_first=True), layers, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, categories)
        self.segment = nn.Parameter(torch.randn(2, hidden) * .02)

    def time_embedding(self, time):
        frequencies = torch.exp(torch.arange(self.hidden//2,device=time.device) * (-math.log(10000)/(self.hidden//2-1)))
        phase = time[:,None] * frequencies[None] * 1000
        return torch.cat([phase.sin(),phase.cos()],dim=1)

    def forward(self, value, time):
        if value.shape[1:] != (self.dimensions,self.categories):
            raise ValueError('Score model belongs to a different architecture encoding')
        encoded = self.operation(value) + self.position[None] + self.time(self.time_embedding(time))[:,None]
        return self.output(self.encoder(encoded))

    def paired_reconstruction_loss(self, first, second, first_mask, second_mask):
        a, b = self.operation(first), self.operation(second)
        a = torch.where(first_mask[:,:,None], torch.zeros_like(a), a)
        b = torch.where(second_mask[:,:,None], torch.zeros_like(b), b)
        tokens = torch.cat([a+self.position+self.segment[0], b+self.position+self.segment[1]],dim=1)
        prediction = self.output(self.encoder(tokens))
        target = torch.cat([first.argmax(-1),second.argmax(-1)],dim=1)
        mask = torch.cat([first_mask,second_mask],dim=1)
        if not mask.any():
            raise ValueError('Paired CATE-style reconstruction requires masked operations')
        return F.cross_entropy(prediction[mask],target[mask])


class DirectedGraphEncoder(nn.Module):
    def __init__(self, categories, hidden=64, layers=2):
        super().__init__()
        self.input = nn.Linear(categories,hidden)
        self.incoming = nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(layers)])
        self.outgoing = nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(layers)])

    def forward(self, operations, adjacency):
        n = operations.shape[1]
        matrix = adjacency.to(operations) + torch.eye(n,device=operations.device,dtype=operations.dtype)
        forward = matrix / matrix.sum(-1,keepdim=True).clamp_min(1)
        backward = matrix.T / matrix.T.sum(-1,keepdim=True).clamp_min(1)
        value = self.input(operations)
        for incoming,outgoing in zip(self.incoming,self.outgoing):
            value = F.silu(incoming(forward @ value)+outgoing(backward @ value))
        return value.mean(1)


class MetaPredictor(nn.Module):
    def __init__(self,categories,dataset_features,hidden=64):
        super().__init__()
        self.graph = DirectedGraphEncoder(categories,hidden)
        self.data = SetDatasetEncoder(dataset_features,hidden)
        self.output = nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,1))

    def forward(self,operations,adjacency,dataset):
        encoded = self.data(dataset)
        if encoded.shape[0] == 1:
            encoded = encoded.expand(operations.shape[0],-1)
        if encoded.shape[0] != operations.shape[0]:raise ValueError('Dataset/architecture batch sizes differ')
        return self.output(torch.cat([self.graph(operations,adjacency),encoded],dim=1)).squeeze(-1)


class SetDatasetEncoder(nn.Module):
    """Class-prototype attention followed by inter-class attention and pooling."""
    def __init__(self,features,hidden,heads=2):
        super().__init__()
        self.projection=nn.Linear(features,hidden)
        self.prototype_query=nn.Parameter(torch.randn(1,1,hidden)*.02)
        self.dataset_query=nn.Parameter(torch.randn(1,1,hidden)*.02)
        self.within=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.between=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.pool=nn.MultiheadAttention(hidden,heads,batch_first=True)

    def forward(self,dataset):
        if isinstance(dataset,dict):values,valid=dataset['values'],dataset['valid']
        else:values,valid=dataset,None
        if values.ndim==1:values=values[None,None,None,:]
        elif values.ndim==2:values=values[:,None,None,:]
        elif values.ndim==3:values=values[None]
        if values.ndim!=4:raise ValueError('Dataset features must have batch/class/sample/feature axes')
        batch,classes,samples,_=values.shape
        if valid is None:valid=torch.ones((batch,classes,samples),dtype=torch.bool,device=values.device)
        elif valid.ndim==2:valid=valid[None]
        if valid.shape!=(batch,classes,samples) or valid.dtype!=torch.bool:raise ValueError('Explicit dataset sample mask shape differs')
        present=valid.any(-1)
        if not present.any(-1).all():raise ValueError('Dataset descriptor has no observed class')
        tokens=self.projection(values).reshape(batch*classes,samples,-1)
        mask=~valid.reshape(batch*classes,samples)
        # A dummy zero key is used only for an absent class; that prototype is
        # removed by the inter-class mask, preventing all-masked attention NaNs.
        safe=mask.clone();safe[mask.all(1),0]=False
        query=self.prototype_query.expand(batch*classes,-1,-1)
        prototypes=self.within(query,tokens,tokens,key_padding_mask=safe,need_weights=False)[0].reshape(batch,classes,-1)
        prototypes=self.between(prototypes,prototypes,prototypes,key_padding_mask=~present,need_weights=False)[0]
        return self.pool(self.dataset_query.expand(batch,-1,-1),prototypes,prototypes,key_padding_mask=~present,need_weights=False)[0][:,0]


def move_descriptor(descriptor,device):
    return {key:value.to(device) for key,value in descriptor.items()} if isinstance(descriptor,dict) else descriptor.to(device)


class GuidedDiffusion:
    def __init__(self,score,predictor,adjacency,*,sigma_min=.01,sigma_max=10.,steps=100,
                 guidance=1.,predictor_std=.1,target_performance=1.,seed=0,prepared=False):
        if not 0 < sigma_min < sigma_max or type(steps) is not int or steps < 1 or predictor_std <= 0:
            raise ValueError('Valid VE noise schedule and predictor standard deviation required')
        self.score,self.predictor,self.adjacency = score,predictor,adjacency
        self.minimum,self.maximum,self.steps = sigma_min,sigma_max,steps
        self.guidance,self.std,self.target = guidance,predictor_std,target_performance
        self.generator = torch.Generator(device=next(score.parameters()).device).manual_seed(seed)
        self.prepared = prepared

    def sigma(self,time):
        return self.minimum * (self.maximum/self.minimum)**time

    def score_training_loss(self,architectures):
        time = torch.rand(architectures.shape[0],device=architectures.device,generator=self.generator)*.999+.001
        noise = torch.randn(architectures.shape,device=architectures.device,generator=self.generator)
        sigma = self.sigma(time)[:,None,None]
        return ((self.score(architectures+sigma*noise,time)*sigma+noise)**2).mean()

    def generate(self,dataset,count):
        if not self.prepared:
            raise ValueError('Prepare and bind actual search-space score/meta-predictor weights before NAS')
        device = next(self.score.parameters()).device
        value = torch.randn((count,self.score.dimensions,self.score.categories),device=device,generator=self.generator)*self.maximum
        self.score.eval();self.predictor.eval()
        step = .999/self.steps
        for index in range(self.steps):
            time = torch.full((count,),1-index*step,device=device)
            coefficient = self.sigma(time)*math.sqrt(2*math.log(self.maximum/self.minimum))
            with torch.no_grad():
                score = self.score(value,time)
            with torch.enable_grad():
                variable = value.detach().requires_grad_(True)
                mean = self.predictor(variable,self.adjacency,move_descriptor(dataset,device))
                log_likelihood = -.5*((mean-self.target)/self.std)**2
                gradient = torch.autograd.grad(log_likelihood.sum(),variable)[0]
            drift = coefficient[:,None,None].square()*(score+self.guidance*gradient)
            noise = torch.randn(value.shape,device=device,generator=self.generator)
            value = (value+step*drift+coefficient[:,None,None]*math.sqrt(step)*noise).detach()
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite guided diffusion architecture sample')
        choices = value.argmax(-1)
        # Detector-space validity: preserve an executable adaptation site.
        for i,row in enumerate(choices):
            if not row[:6].any():
                row[(value[i,:6,1]-value[i,:6,0]).argmax()] = 1
        return [tuple(map(int,row.tolist())) for row in choices]

    def state_dict(self):
        return {'score':self.score.state_dict(),'predictor':self.predictor.state_dict(),
                'generator':self.generator.get_state(),'prepared':self.prepared}

    def load_state_dict(self,state):
        self.score.load_state_dict(state['score']);self.predictor.load_state_dict(state['predictor'])
        self.generator.set_state(state['generator']);self.prepared=state['prepared']


def prepare_generator(generator,architectures,dataset_features,performances,*,roles,
                      paired_architectures,paired_masks,iterations,learning_rate,
                      evidence_scope='software_fixture_NOT_AP'):
    """Explicit offline preparation; SOURCE/DEV labels only, no hidden download."""
    if any(role not in ('SOURCE','DEV') for role in roles) or len(roles)!=len(architectures):
        raise ValueError('Generation preparation requires explicitly identified SOURCE/DEV examples')
    feature_values=dataset_features['values'] if isinstance(dataset_features,dict) else dataset_features
    if architectures.shape[0] != len(performances) or feature_values.shape[0] != len(performances):
        raise ValueError('Aligned architecture/dataset/performance triples required')
    if type(iterations) is not int or iterations < 1 or learning_rate <= 0:
        raise ValueError('Positive offline preparation steps and learning rate required')
    if evidence_scope not in ('software_fixture_NOT_AP','native_detector_preparation'):
        raise ValueError('Explicit native preparation or software fixture evidence scope required')
    parameters=list(generator.score.parameters())+list(generator.predictor.parameters())
    optimizer=torch.optim.Adam(parameters,lr=learning_rate)
    losses=[]
    generator.score.train();generator.predictor.train()
    for phase in ('paired_encoder','score_matching','meta_prediction'):
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            if phase=='paired_encoder':
                loss=generator.score.paired_reconstruction_loss(architectures,paired_architectures,*paired_masks)
            elif phase=='score_matching':loss=generator.score_training_loss(architectures)
            else:
                prediction=generator.predictor(architectures,generator.adjacency,dataset_features)
                loss=F.mse_loss(prediction,performances)
            if not torch.isfinite(loss):raise ValueError('Nonfinite diffusion preparation objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            optimizer.step()
            losses.append({'phase':phase,'loss':float(loss.detach())})
    generator.prepared=True
    return {'iterations_per_phase':iterations,'total_optimizer_steps':3*iterations,'losses':losses,'preparation_examples':len(roles),
            'scope':'detector_search_space_preparation','evidence_scope':evidence_scope,'author_checkpoint':False}
