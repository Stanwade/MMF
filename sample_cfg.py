import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from typing import Optional
from utils import plot_imgs, calculate_ssim

import warnings
warnings.filterwarnings("ignore")

from DiffusionModel.diffusion import DiffusionModel
from ReconstructionModel.model import ReconstructionModel
from datasets import create_dataloader

def sample_cfg(diffusion_model_dir:str, 
               batch_size:int,
               dataset_type:str = 'imgnet32', 
               img_channels:int = 3,
               cfg_weight:float = 3.0, 
               skip_to:int = 100, 
               reconstruction_model_dir:str = None,
               img_dir:str = 'img',
               **kwargs):
    
    if reconstruction_model_dir is not None:
        model = DiffusionModel.load_from_checkpoint(diffusion_model_dir, 
                                                    cfg=cfg_weight,
                                                    reconstruction_model_dir=reconstruction_model_dir,
                                                    map_location='cuda')
    else:
        model = DiffusionModel.load_from_checkpoint(diffusion_model_dir,
                                                    cfg=cfg_weight,
                                                    map_location='cuda')
    model.eval()
    
    r_model = ReconstructionModel.load_from_checkpoint(reconstruction_model_dir)
    
    _, valid_dataset, _, _ = create_dataloader(dataset_type=dataset_type, need_datasets=True)
    
    valid_loader = data.DataLoader(valid_dataset, 
                                   batch_size=batch_size * img_channels, 
                                   shuffle=False)
    inputs_rgb = []
    labels_rgb = []
    outs_recon_rgb = []
    outs_diffusion_rgb = []
    
    inputs, labels = next(iter(valid_loader))
    
    outputs_recon = r_model(inputs)
    
    outputs_diffusion = model.sample_backward(inputs, skip_to=skip_to)
    
    
    for i in range(batch_size):
        # input i shape (1,height,width)
        inputs_rgb.append(torch.unsqueeze(torch.cat([inputs[img_channels*i],
                                                         inputs[img_channels*i+1],
                                                         inputs[img_channels*i+2]]),
                                         axis=0) / 255)
        
        # labels i shape (1,height,width)
        labels_rgb.append(torch.unsqueeze(torch.cat([labels[img_channels*i],
                                                         labels[img_channels*i+1],
                                                         labels[img_channels*i+2]]),
                                         axis=0))
        
        # outputs_recon i shape (1,height,width)
        outs_recon_rgb.append(torch.unsqueeze(torch.cat([outputs_recon[img_channels*i],
                                                             outputs_recon[img_channels*i+1],
                                                             outputs_recon[img_channels*i+2]]),
                                             axis=0))
        
        # outputs_diffusion i shape (1,height,width)
        outs_diffusion_rgb.append(torch.unsqueeze(torch.cat([outputs_diffusion[img_channels*i],
                                                                 outputs_diffusion[img_channels*i+1],
                                                                 outputs_diffusion[img_channels*i+2]]),
                                                 axis=0))
    
        if i == batch_size - 1:
            # (batch_size, channels, height, width) -> (batch_size, height, width, channels)
            inputs_rgb = torch.cat(inputs_rgb, axis=0).permute(0, 2, 3, 1)
            labels_rgb = torch.cat(labels_rgb, axis=0).permute(0, 2, 3, 1)
            outs_recon_rgb = torch.cat(outs_recon_rgb, axis=0).permute(0, 2, 3, 1)
            outs_diffusion_rgb = torch.cat(outs_diffusion_rgb, axis=0).permute(0, 2, 3, 1)

    ssim_list = []

    for i in range(batch_size):
        ssim = calculate_ssim(outs_diffusion_rgb[i], labels_rgb[i])
        ssim_list.append(ssim)
    
    plot_imgs(inputs_rgb, name='01_inputs',cmap=None, dir=img_dir)
    plot_imgs(labels_rgb, name='02_label',cmap=None, dir=img_dir)
    plot_imgs(outs_recon_rgb, name='03_out_recon',cmap=None, dir=img_dir)
    plot_imgs(outs_diffusion_rgb, name='04_out_diffusion',cmap=None, dir=img_dir, str_list=ssim_list)
    
    
    return outs_diffusion_rgb

if __name__ == '__main__':
    sample_cfg(diffusion_model_dir='./DiffusionModel/ckpts_imgnet32/epoch=59-val_loss=0.0266.ckpt',
               batch_size=5,
               dataset_type='imgnet32',
               img_channels=3,
               cfg_weight=3.0,
               skip_to=100,
               reconstruction_model_dir=None,
               img_dir='img')