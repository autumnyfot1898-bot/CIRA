"""Explicit algorithm constructors; unavailable methods never fall back to CIRA."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch

from .dpfl import DPFL
from .dspodfl import DSpodFL
from .dadpfl import DADPFL
from .dcfcl import DCFCL
from .pfeddil import PFedDIL
from .dflmoe import DFLMoE
from .perfedsde import PerFedSDENAS
from .demodcl import DeModCL
from .nas_diffusion import GraphScore,MetaPredictor,GuidedDiffusion

IMPLEMENTED={'DPFL':DPFL,'DSpodFL':DSpodFL,'DA-DPFL':DADPFL,'DCFCL':DCFCL,
             'pFedDIL':PFedDIL,'dFLMoE':DFLMoE,'PerFedSDE-NAS':PerFedSDENAS,'DeModCL':DeModCL}


def bound_torch_asset(binding):
    if not isinstance(binding,dict) or not binding.get('path') or not binding.get('sha256'):
        raise ValueError('Fill an actual asset path and SHA-256 in local configuration')
    path=Path(binding['path'])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:
        raise ValueError('Bound asset is missing or changed')
    return torch.load(path,map_location='cpu',weights_only=True)


def load_diffusion(binding,device='cpu'):
    saved=bound_torch_asset(binding)
    if saved.get('schema')!='cira.detector_diffusion.v1' or saved.get('preparation',{}).get('scope')!='detector_search_space_preparation':
        raise ValueError('Bind the prepared detector-specific diffusion artifact, not arbitrary weights')
    if saved['preparation'].get('evidence_scope')!='native_detector_preparation':
        raise ValueError('A software fixture cannot become a native prepared-generation asset')
    score=GraphScore(**saved['score_config']).to(device)
    predictor=MetaPredictor(**saved['predictor_config']).to(device)
    generator=GuidedDiffusion(score,predictor,saved['adjacency'].to(device),**saved['diffusion_config'])
    generator.load_state_dict(saved['generator'])
    if not generator.prepared:raise ValueError('Diffusion preparation is incomplete')
    return generator


def create(method,models,options,assets=None):
    if method not in IMPLEMENTED:raise ValueError('Unknown external detector algorithm')
    options=dict(options)
    if method=='PerFedSDE-NAS':
        assets=assets or {}
        options['generator']=load_diffusion(assets.get('diffusion'),assets.get('diffusion_device','cpu'))
        descriptor=bound_torch_asset(assets.get('dataset_descriptors'))
        if descriptor.get('role')!='D' or descriptor.get('schema')!='cira.nas_dataset_descriptors.v1':
            raise ValueError('Bind D-only dataset descriptors for the actual clients')
        options['dataset_descriptors']=descriptor['descriptors']
    result=IMPLEMENTED[method](models,**options)
    if method=='PerFedSDE-NAS':result.descriptors_by_stage=descriptor.get('by_stage',{})
    return result
