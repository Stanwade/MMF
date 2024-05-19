import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from typing import Optional, Union, List
from utils import create_act, create_norm
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width, base_number:int = 2, min_num_freq:int = 4):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        # Create positional encoding for x and y coordinates
        self.pos_x = torch.linspace(0, 1, steps=width).reshape(1, 1, 1, -1).repeat(1, 1, height, 1)
        self.pos_y = torch.linspace(0, 1, steps=height).reshape(1, 1, -1, 1).repeat(1, 1, 1, width)
        # Initialize frequencies
        self.freqs = torch.pow(base_number, torch.arange(0, max(channels//4 ,min_num_freq), 1).float() / (max(channels//2, 1)))
        
    def forward(self, x:torch.Tensor):
        x = x.view(x.shape[0], -1, self.height, self.width)
        batch_size = x.shape[0]
        self.pos_x = self.pos_x.repeat(batch_size,1,1,1)
        self.pos_y = self.pos_y.repeat(batch_size,1,1,1)
        
        sin_x = torch.sin(torch.einsum("bchw, f -> bcfhw", self.pos_x, self.freqs).reshape(x.shape[0], -1, x.shape[2], x.shape[3]))
        cos_x = torch.cos(torch.einsum("bchw, f -> bcfhw", self.pos_x, self.freqs).reshape(x.shape[0], -1, x.shape[2], x.shape[3]))
        sin_y = torch.sin(torch.einsum("bchw, f -> bcfhw", self.pos_y, self.freqs).reshape(x.shape[0], -1, x.shape[2], x.shape[3]))
        cos_y = torch.cos(torch.einsum("bchw, f -> bcfhw", self.pos_y, self.freqs).reshape(x.shape[0], -1, x.shape[2], x.shape[3]))
        return torch.cat([x, sin_x, cos_x, sin_y, cos_y], dim=1).squeeze(1)

    def calculate_channel_dim(self):
        return len(self.freqs) * 4 + self.channels

class FullyConnectedNetwork(nn.Module):
    def __init__(self,
                 in_img_shape: torch.Tensor,
                 out_img_shape: torch.Tensor,
                 mid_lengths = [512, 256],
                 norm_type="batchnorm",
                 activation="mish",
                 img_size = 16,
                 with_pe = False):
        super(FullyConnectedNetwork, self).__init__()
        mid_lengths = mid_lengths + [out_img_shape[0] * out_img_shape[1] * out_img_shape[2]]
        self.mid_lengths = mid_lengths
        self.flatten = nn.Flatten()
        self.layers_num = len(self.mid_lengths)
        self.blocks = nn.ModuleList([])
        self.norm = create_norm(norm_type=norm_type, in_img_shape=in_img_shape)
        self.act = create_act(activation=activation)
        self.img_size = img_size
        
        
        for i in range(self.layers_num):
            if i == 0:
                self.blocks.append(self.norm)
                self.blocks.append(self.flatten)
                self.blocks.append(nn.Linear(in_img_shape[0] * in_img_shape[1] * in_img_shape[2],
                                            mid_lengths[i]))
                self.blocks.append(self.act)
            else:
                self.blocks.append(nn.Linear(mid_lengths[i-1], mid_lengths[i]))
                self.blocks.append(self.act)
        
        if with_pe:
            self.pe = PositionalEncoding2D(out_img_shape[0], out_img_shape[1], out_img_shape[2])
            pe_out_channels = self.pe.calculate_channel_dim()
            self.blocks.append(self.pe)
            self.blocks.append(self.flatten)
            self.blocks.append(nn.Linear(pe_out_channels * out_img_shape[1] * out_img_shape[2], out_img_shape[1] * out_img_shape[2]))
            self.blocks.append(self.act)
            
        
    def forward(self, x):
        # print(self.blocks)
        for block in self.blocks:
            x = block(x)
        return x.view(-1, 1, self.img_size, self.img_size)
    
    
if __name__ == '__main__':
    test_x = torch.randn(32, 1, 100, 100)
    model = FullyConnectedNetwork([1,100,100], [1,16,16], mid_lengths=[], with_pe=True)
    out = model(test_x)
    print(out.size())
        
        