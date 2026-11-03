"""Explicit native YOLO detector search space for the PerFedSDE-NAS port.

Six residual bottleneck sites and two bottleneck ratios use shared maximum-width
weights. This is a detector-specific search space; it is not an author NASBench,
MobileNet, DARTS or NASViT checkpoint. There is no CIRA controller/state here.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        width = max(1, math.ceil(channels / 8))
        self.down = nn.Conv2d(channels, width, 1)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.up = nn.Conv2d(width, channels, 1)
        self.channels = channels

    def forward(self, value, ratio):
        width = min(self.down.out_channels, max(1, math.ceil(self.channels / ratio)))
        hidden = F.silu(F.conv2d(value, self.down.weight[:width], self.down.bias[:width]))
        hidden = F.silu(F.conv2d(hidden, self.depthwise.weight[:width], self.depthwise.bias[:width], padding=1, groups=width))
        residual = F.conv2d(hidden, self.up.weight[:, :width], self.up.bias)
        return value + residual


class DetectorSupernet(nn.Module):
    dimensions, categories = 7, 2
    def __init__(self, base, port_indices, image_size=640):
        super().__init__()
        if len(port_indices) != 6 or len(set(port_indices)) != 6:
            raise ValueError('Declare six distinct native feature ports')
        self.base, self.port_indices = base, tuple(port_indices)
        observed, handles = {}, []
        for index in self.port_indices:
            handles.append(base.model[index].register_forward_hook(lambda m,a,o,i=index: observed.__setitem__(i,o.shape)))
        modes = [(module, module.training) for module in base.modules()]
        base.eval()
        try:
            with torch.no_grad():
                base(torch.zeros(1,3,image_size,image_size,device=next(base.parameters()).device))
        finally:
            for handle in handles:
                handle.remove()
            for module, mode in modes:
                module.training = mode
        if any(index not in observed or len(observed[index]) != 4 for index in self.port_indices):
            raise ValueError('Each search-space port must output one feature map')
        self.bottlenecks = nn.ModuleList([Bottleneck(observed[index][1]) for index in self.port_indices]).to(next(base.parameters()).device)
        self.architecture = (1,1,1,1,1,1,0)

    @property
    def model(self):
        return self.base.model

    def set_architecture(self, architecture):
        architecture = tuple(architecture)
        if len(architecture) != self.dimensions or any(type(x) is not int or x not in (0,1) for x in architecture) or not any(architecture[:6]):
            raise ValueError('Seven binary choices and at least one enabled feature site required')
        self.architecture = architecture

    def forward(self, images):
        handles = []
        ratio = 16 if self.architecture[-1] == 0 else 8
        for position, index in enumerate(self.port_indices):
            if self.architecture[position]:
                layer = self.bottlenecks[position]
                handles.append(self.base.model[index].register_forward_hook(lambda m,a,o,layer=layer: layer(o,ratio)))
        try:
            return self.base(images)
        finally:
            for handle in handles:
                handle.remove()

    def loss(self, batch, preds=None):
        return self.base.loss(batch, preds=preds)

