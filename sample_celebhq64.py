import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from typing import Optional
from utils import plot_imgs, calculate_ssim
from tqdm import trange
import cv2

import warnings
warnings.filterwarnings("ignore")

from DiffusionModel.diffusion import DiffusionModel
from ReconstructionModel.model import ReconstructionModel
from datasets import create_dataloader

def sample_celebhq64(
        diffusion_model_dir:str,
        batch_size:int,
        dataset_type:str = 'celebHQ64',
        img_channels:int = 3):
    print('loading diffusion models...')
    model = DiffusionModel.load_from_checkpoint(diffusion_model_dir, map_location='cuda')
    # sample from noise
    print('sampling from noise...')
    noise = torch.randn(batch_size, img_channels, 64, 64).to('cuda')
    samples = model.sample_backward(noise,model.device,cfg_act=False,skip_to=1000)*255
    print(torch.max(samples))
    print(torch.min(samples))
    # save to imgs
    for i in range(batch_size):
        cv2.imwrite(f'./imgs/out_{i}.jpg',samples[i].permute(1,2,0).to('cpu').numpy())

if __name__ == "__main__":
    sample_celebhq64('/home/wty/mmf_-demo/DiffusionModel/ckpts_celebHQ64_wo_condition/last.ckpt',5)