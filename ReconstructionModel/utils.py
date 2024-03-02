import torch
import torch.nn as nn

def create_norm(norm_type: str, in_img_shape: torch.Tensor):
    if norm_type == "batchnorm":
        norm = nn.BatchNorm2d(in_img_shape[0])
    elif norm_type == "instancenorm":
        norm = nn.InstanceNorm2d(in_img_shape[0])
    elif norm_type == "layernorm":
        norm = nn.LayerNorm(in_img_shape)
    elif norm_type == "none":
        norm = nn.Identity()
    else:
        raise NotImplementedError('Norm type not implemented')
    return norm

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