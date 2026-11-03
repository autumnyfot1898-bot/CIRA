"""Explicit offline detector-search generation-model preparation command."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .factory import bound_torch_asset
from .nas_diffusion import GraphScore,MetaPredictor,GuidedDiffusion,prepare_generator


def run(spec):
    if spec.get('schema')!='cira.diffusion_preparation_spec.v1':raise ValueError('Explicit preparation specification required')
    output=Path(spec.get('output',''))
    if not spec.get('output'):raise ValueError('Fill the private preparation output path')
    if output.exists():raise FileExistsError('Preparation output exists; choose a new artifact path')
    data=bound_torch_asset(spec.get('triples'))
    if data.get('schema')!='cira.native_architecture_triples.v1' or data.get('metric')!='AP50:95' or data.get('score_units')!='fraction':
        raise ValueError('Bind actual detector architecture/dataset/AP triples with explicit metric units')
    if data.get('scope')!='native_detector_preparation':raise ValueError('Numerical-policy or synthetic triples cannot become native preparation evidence')
    device=spec.get('device')
    if not device:raise ValueError('Fill the actual preparation device')
    score=GraphScore(**spec['score_config']).to(device)
    predictor=MetaPredictor(**spec['predictor_config']).to(device)
    adjacency=data['adjacency'].to(device)
    generator=GuidedDiffusion(score,predictor,adjacency,**spec['diffusion_config'])
    move=lambda value:{k:v.to(device) for k,v in value.items()} if isinstance(value,dict) else value.to(device)
    report=prepare_generator(generator,data['architectures'].to(device),move(data['dataset_features']),data['performances'].to(device),
        roles=data['roles'],paired_architectures=data['paired_architectures'].to(device),
        paired_masks=tuple(value.to(device) for value in data['paired_masks']),
        iterations=spec['iterations_per_phase'],learning_rate=spec['learning_rate'],evidence_scope='native_detector_preparation')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.pending.pt')
    artifact={'schema':'cira.detector_diffusion.v1','score_config':spec['score_config'],
        'predictor_config':spec['predictor_config'],'diffusion_config':spec['diffusion_config'],
        'adjacency':adjacency.cpu(),'generator':generator.state_dict(),'preparation':report,
        'triples_sha256':spec['triples']['sha256'],'preparation_spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()}
    torch.save(artifact,temporary);temporary.replace(output)
    report={**report,'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'formal_qualification':False}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    args=parser.parse_args();report=run(json.loads(Path(args.spec).read_text(encoding='utf-8')))
    print(json.dumps({key:report[key] for key in ('iterations_per_phase','total_optimizer_steps','output_sha256')}))


if __name__=='__main__':main()
