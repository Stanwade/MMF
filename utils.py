import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Union
from inspect import isfunction
from torch.utils.data import DataLoader
from datasets import MNISTDataset, MMFDataset, MMFGrayScaleDataset, MMFMNISTDataset


def plot_imgs(inputs,name:str, dir:str='imgs', figsize = (16,16), str_list: List[str] = None):
    fig, axes = plt.subplots(nrows=1, ncols=inputs.size(0), figsize=figsize)
    for idx in range(inputs.size(0)):
        axes[idx].imshow(inputs[idx].squeeze().numpy(), cmap='gray')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        
        if str_list:
            if len(str_list) != inputs.size(0):
                raise ValueError('plot error: len(str_list) != inputs.size(0)')
            
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

def create_dataloader(dataset_type: str,
                      root: str='./datasets/',
                      target_pipeline = None,
                      batch_size: int = 64,
                      num_workers: int = 96,
                      need_datasets: bool = False):
    if dataset_type == 'MNIST':
        train_dataset = MNISTDataset(root=root, train=True, transform=target_pipeline)
        validation_dataset = MNISTDataset(root=root, train=False, transform=target_pipeline)
        
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    elif dataset_type == 'MMF':
        # Load data
        train_dataset = MMFDataset(root=root,
                                train=True,
                                target_transform=target_pipeline)
        validation_dataset = MMFDataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif dataset_type =='MMFGrayscale':
        # Load data
        train_dataset = MMFGrayScaleDataset(root=root,
                                            train=True,
                                            target_transform=target_pipeline)
        validation_dataset = MMFGrayScaleDataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif dataset_type == 'MMFMNIST':
        train_dataset = MMFMNISTDataset(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMFMNISTDataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    else:
        raise NotImplementedError(f"dataset type {dataset_type} doesn't exist!")
    
    if need_datasets:
        return train_dataset, validation_dataset, train_loader, validation_loader
    else:
        return train_loader, validation_loader
    

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

