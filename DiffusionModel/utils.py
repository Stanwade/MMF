from inspect import isfunction
import torch.nn as nn
import matplotlib.pyplot as plt
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
    elif norm_type == "groupnorm":
        return nn.GroupNorm(32, in_channels)
    elif norm_type == 'none':
        return nn.Identity()
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
    elif act_type == "silu":
        return nn.SiLU()
    else:
        raise NotImplementedError


def plot_imgs(inputs,name:str, dir:str='imgs', figsize = (16,16)):
    fig, axes = plt.subplots(nrows=1, ncols=inputs.size(0), figsize=figsize)
    for idx in range(inputs.size(0)):
        axes[idx].imshow(inputs[idx].squeeze().numpy(), cmap='gray')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{dir}/{name}.png')
    
def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d