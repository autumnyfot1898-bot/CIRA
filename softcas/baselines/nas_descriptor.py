"""D-only region descriptors for the detector-specific NAS meta-predictor."""
from __future__ import annotations
from pathlib import Path
import hashlib
import random

import torch
from torch.nn import functional as F


def resnet18_encoder(checkpoint,device='cpu'):
    from torchvision.models import resnet18
    path=Path(checkpoint)
    if not path.is_file():raise FileNotFoundError('Bind the actual ResNet-18 encoder checkpoint')
    model=resnet18(weights=None)
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(state.get('state_dict',state))
    model.fc=torch.nn.Identity()
    return model.to(device).eval().requires_grad_(False),hashlib.sha256(path.read_bytes()).hexdigest()


def region_descriptor(data,encoder,per_class,seed,classes=8):
    """Equal count per observed class; explicit masks retain absent classes.

    The classification paper uses labelled images. Detection adaptation uses
    labelled object crops from D only, preserving the paper's category grouping.
    Repeated crops fill short class groups; they are not independent samples.
    """
    if type(per_class) is not int or per_class<1:raise ValueError('Positive per-class sample count required')
    rng=random.Random(seed)
    pools=[[] for _ in range(classes)]
    for batch in data.require('D'):
        for index,box,cls in zip(batch['batch_idx'].tolist(),batch['bboxes'],batch['cls'].flatten().tolist()):
            cls=int(cls)
            if not 0<=cls<classes:raise ValueError('Descriptor class outside fixed vocabulary')
            image=batch['img'][index]
            height,width=image.shape[-2:]
            cx,cy,w,h=box.tolist()
            left,right=max(0,int((cx-w/2)*width)),min(width,max(1,int((cx+w/2)*width)))
            top,bottom=max(0,int((cy-h/2)*height)),min(height,max(1,int((cy+h/2)*height)))
            if right<=left or bottom<=top:continue
            pools[cls].append(image[:,top:bottom,left:right].detach())
    if not any(pools):raise ValueError('D descriptor requires at least one labelled object crop')
    device=next(encoder.parameters()).device
    values=torch.zeros(classes,per_class,512,device=device)
    valid=torch.zeros(classes,per_class,dtype=torch.bool,device=device)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    encoder.eval()
    with torch.no_grad():
        for cls,pool in enumerate(pools):
            if not pool:continue
            indices=rng.sample(range(len(pool)),per_class) if len(pool)>=per_class else rng.choices(range(len(pool)),k=per_class)
            crops=torch.cat([F.interpolate(pool[index][None].to(device),size=(224,224),mode='bilinear',align_corners=False) for index in indices])
            features=encoder((crops-mean)/std)
            if features.shape!=(per_class,512) or not torch.isfinite(features).all():raise ValueError('Pinned ResNet-18 feature shape changed')
            values[cls]=features;valid[cls]=True
    return {'values':values,'valid':valid}
