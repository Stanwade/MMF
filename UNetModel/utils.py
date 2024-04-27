from pytorch_lightning import LightningModule, Trainer
import torch
import torch.nn as nn
from typing import Mapping, Any
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback
from inspect import isfunction

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def create_norm(in_channels, norm_type="batchnorm"):
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(in_channels)
    elif norm_type == "instancenorm":
        return nn.InstanceNorm2d(in_channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise NotImplementedError("Norm type not implemented")

def create_act(activation: str):
    if activation == "mish":
        act = nn.Mish()
    elif activation == "relu":
        act = nn.ReLU()
    elif activation == "lrelu":
        act = nn.LeakyReLU()
    elif activation == "silu":
        act = nn.SiLU()
    elif activation == "none":
        act = nn.Identity()
    else:
        raise NotImplementedError("Activation not implemented")
    return act