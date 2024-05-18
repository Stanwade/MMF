import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Union
from inspect import isfunction
from torch.utils.data import DataLoader
from datasets import MNISTDataset, MMFDataset, MMFGrayScaleDataset, MMFMNISTDataset, MMFMNISTDataset_grayscale


def plot_imgs(inputs,name:str, dir:str='imgs', figsize = (16,16), str_list: List[str] = None, cmap='gray'):
    fig, axes = plt.subplots(nrows=1, ncols=inputs.size(0), figsize=figsize)
    for idx in range(inputs.size(0)):
        axes[idx].imshow(inputs[idx].squeeze().numpy(), cmap=cmap)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        
        if str_list:
            if len(str_list) != inputs.size(0):
                raise ValueError(f'plot error: len(str_list) != inputs.size(0), len(str_list)={len(str_list)}, inputs.size(0)={inputs.size(0)}')
            
            axes[idx].text(0.5,
                           -0.08,
                           f'SSIM={float(str_list[idx]):.4f}',
                           color='black',
                           ha='center',
                           va = 'bottom',
                           transform=axes[idx].transAxes)
            
    plt.tight_layout()
    plt.savefig(f'{dir}/{name}.png')
    
def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def calculate_ssim(x, y, k1 = 0.01, k2 = 0.03, L = 255):
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    elif not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    
    assert x.shape == y.shape, f'input tensors must have the same shape, x shape is {x.shape}, y shape is {y.shape}'
    
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    mu_x = x.mean(dim=[0, 1, 2], keepdim=True)
    mu_y = y.mean(dim=[0, 1, 2], keepdim=True)
    mu_x_mu_y = mu_x * mu_y
    sigma_x_sq = x.var(dim=[0, 1, 2], unbiased=False, keepdim=True)
    sigma_y_sq = y.var(dim=[0, 1, 2], unbiased=False, keepdim=True)
    sigma_x_y = torch.mean(x * y, dim=[0, 1, 2], keepdim=True) - mu_x_mu_y
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    l = (2 * mu_x_mu_y + C1) / (mu_x_sq + mu_y_sq + C1)
    c = (2 * sigma_x_y + C2) / (sigma_x_sq + sigma_y_sq + C2)

    return l * c

