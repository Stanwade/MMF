from inspect import isfunction
import torch.nn as nn

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def create_norm(in_channels, norm_type="batchnorm"):
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(in_channels)
    elif norm_type == "instancenorm":
        return nn.InstanceNorm2d(in_channels)
    else:
        raise NotImplementedError

def create_activation(act_type="relu"):
    if act_type == "relu":
        return nn.ReLU()
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "mish":
        return nn.Mish()
    elif act_type == "lrelu":
        return nn.LeakyReLU()
    else:
        raise NotImplementedError