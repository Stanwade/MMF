import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import numpy as np
import random
import math
import os
import sys
import time

from UNetModel.utils import create_act, create_norm, default
from typing import Union, List

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 bias=False,
                 activation="lrelu",
                 norm_type="batchnorm"):
        super(ResBlock, self).__init__()
        self.norm1 = create_norm(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm2 = create_norm(out_channels, norm_type)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        if in_channels != out_channels or stride != 1:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connect = nn.Identity()

        
        self.act = create_act(activation)
        
    def forward(self, x):
        # print(x.size())
        h = self.norm1(x)
        h = self.act(h)                     # b, c, h, w
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.skip_connect(x)

class SelfAttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 norm_type="batchnorm",):
        super(SelfAttnBlock, self).__init__()
        self.norm = create_norm(dim, norm_type)
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.k = nn.Conv2d(dim, dim, kernel_size=1)
        self.v = nn.Conv2d(dim, dim, kernel_size=1)
        self.out = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, x):
        
        n, c, h, w = x.shape
        norm_x = self.norm(x)
        q = self.q(norm_x)
        k = self.k(norm_x)
        v = self.v(norm_x)
        
        # n, c, h, w -> n, h*w, c
        q = q.reshape(n, c, h*w).permute(0, 2, 1)
        
        # n c h w -> n c h*w
        k = k.reshape(n, c, h*w)
        
        qk = torch.matmul(q, k)/math.sqrt(c)
        qk = F.softmax(qk, dim=-1)
        # qk: n, h*w, h*w
        
        v = v.reshape(n, c, h*w).permute(0, 2, 1)
        res = torch.bmm(qk, v)
        res = res.reshape(n, c, h, w)
        res = self.out(res)
        
        return x + res
    
class ResAttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_attn=False,
                 norm_type="batchnorm",
                 activation="lrelu"):
        super(ResAttnBlock, self).__init__()
        self.res_block = ResBlock(in_channels,out_channels,norm_type=norm_type, activation=activation)
        if with_attn:
            self.attn_block = SelfAttnBlock(out_channels, norm_type=norm_type)
        else:
            self.attn_block = nn.Identity()
            
    def forward(self, x):
        x = self.res_block(x)
        x = self.attn_block(x)
        return x
    
class ResAttnBlockMiddle(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_attn=False,
                 norm_type="batchnorm",
                 activation="lrelu"):
        super(ResAttnBlockMiddle, self).__init__()
        self.res_block1 = ResBlock(in_channels,
                                  out_channels,
                                  norm_type=norm_type,
                                  activation=activation)
        self.res_block2 = ResBlock(out_channels,
                                  out_channels,
                                  norm_type=norm_type,
                                  activation=activation)
        if with_attn:
            self.attn_block = SelfAttnBlock(out_channels, norm_type=norm_type)
        else:
            self.attn_block = nn.Identity()
            
    def forward(self, x):
        x = self.res_block1(x)
        x = self.attn_block(x)
        x = self.res_block2(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv=False):
        super(Downsample, self).__init__()
        out_channels = default(out_channels, in_channels)
        if use_conv:
            self.op = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.op = nn.MaxPool2d(2, 2)
            
    def forward(self, x: torch.Tensor):
        return self.op(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv=False, scale_factor=2):
        super(Upsample, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.use_conv = use_conv
        
        out_channels = default(out_channels, in_channels)
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        
    def forward(self, x: torch.Tensor):
        x = x.float()
        if self.use_conv:
            return self.conv(self.up(x))
        else:
            return self.up(x)

class UNetLevel(nn.Module):
    def __init__(self,
                 blocks: int,
                 inout_channels: int,
                 mid_channels: int,
                 mid_block: nn.Module,
                 with_attn: bool = False,
                 norm_type="batchnorm",
                 activation="lrelu",
                 down_up_sample: bool = True
                 ):
        super(UNetLevel, self).__init__()
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        
        for _ in range(blocks):
            self.downs.append(ResAttnBlock(inout_channels,
                                           mid_channels,
                                           with_attn=with_attn,
                                           norm_type=norm_type,
                                           activation=activation))
            self.ups.insert(0, ResAttnBlock(mid_channels*2,
                                            inout_channels,
                                            with_attn=with_attn,
                                            norm_type=norm_type,
                                            activation=activation))
            inout_channels = mid_channels
        
        if down_up_sample:
            self.down = Downsample(mid_channels)
            self.up = Upsample(mid_channels)
        else:
            self.down = nn.Identity()
            self.up = nn.Identity()
        
        self.mid_block = mid_block
        
    def forward(self, x: torch.Tensor):
        hs = []
        for down in self.downs:
            x = down(x)
            hs.append(x)

        x = self.down(x)
        x = self.mid_block(x)
        x = self.up(x)
        
        for up in self.ups:
            h = hs.pop()
            x = up(torch.concat((h,x), dim=1))
        
        return x
    
class UNet(nn.Module):
    def __init__(
        self,
        blocks:int,
        img_channels:int,
        base_channels:int = 64,
        ch_mult: list = [1,2,4,4],
        norm_type="batchnorm",
        activation="lrelu",
        with_attn: Union[bool, List[bool]] = False,
        down_up_sample: bool = True
    ):
        super(UNet, self).__init__()
        
        # dims = [base_channels, base_channels, base_channels*2, base_channels*4, base_channels*4]
        
        if isinstance(with_attn, list) and len(with_attn) != len(ch_mult):
            raise ValueError("Attention: 'with_attn' must be a list of same length as 'ch_mult'!")
        
        self.levels = len(ch_mult)
        dims = [base_channels] + [int(base_channels * mult) for mult in ch_mult]
        self.with_attn = [with_attn] * self.levels if isinstance(with_attn, bool) else with_attn
        
        in_out = list(zip(dims[:-1], dims[1:], self.with_attn[:-1]))
        self.img_channels = img_channels
        
        
        
        self.in_proj = nn.Conv2d(img_channels, base_channels, 3, padding=1, stride=1)
        self.out_proj = nn.Sequential(
            # nn.GroupNorm(32, base_channels),
            # nn.Mish(),
            nn.BatchNorm2d(base_channels),
            nn.Conv2d(base_channels, img_channels, 3, padding=1, stride=1)
        )
        
        
        
        # build unet
        # begin with middle block
        if self.with_attn[-1]:
            now_blocks = ResAttnBlockMiddle(base_channels*ch_mult[-1],
                                            base_channels*ch_mult[-1],
                                            with_attn=self.with_attn[-1],
                                            norm_type=norm_type,
                                            activation=activation)
        else:
            now_blocks = ResBlock(base_channels*ch_mult[-1], base_channels*ch_mult[-1])
        for inout_ch, mid_ch, attn in reversed(in_out):
            now_blocks = UNetLevel(blocks,
                                   inout_ch,
                                   mid_ch,
                                   mid_block=now_blocks,
                                   with_attn=attn,
                                   norm_type=norm_type,
                                   activation=activation,
                                   down_up_sample=down_up_sample)
        
        self.unet = now_blocks
    
    def forward(self, x: torch.Tensor):
        
        x = self.in_proj(x)
        return self.out_proj(self.unet(x))
    
if __name__ == "__main__":
    test_x = torch.randn(1, 1, 32, 32).to('cuda')
    
    unet_config = {
        'blocks': 2,
        'img_channels': 1,
        'base_channels': 1,
        'ch_mult': [1,2,6,6],
        'norm_type': 'batchnorm',
        'activation': 'lrelu',
        'with_attn': [False,False,False,False,False,True],
        'down_up_sample': False
    }
    
    unet = UNet(**unet_config).to('cuda')
    test_out = unet(test_x)
    print(f'hahaha: {torch.cuda.memory_allocated()}')
    print(test_out.shape)
    
    from torchinfo import summary
    summary(unet, input_size=(1, 1, 32, 32), depth=4)
    
    # print(unet)

    