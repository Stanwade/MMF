import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from typing import Optional, Union, List
from ReconstructionModel.utils import create_act, create_norm

class FullyConnectedNetwork(nn.Module):
    def __init__(self,
                 in_img_shape: torch.Tensor,
                 out_img_shape: torch.Tensor,
                 mid_lengths = [512, 256],
                 norm_type="batchnorm",
                 activation="mish",
                 img_size = 16):
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
        
        
    def forward(self, x):
        # print(self.blocks)
        for block in self.blocks:
            x = block(x)
        return x.view(-1, 1, self.img_size, self.img_size)
    
    
if __name__ == '__main__':
    test_x = torch.randn(32, 1, 100, 100)
    model = FullyConnectedNetwork([1,100,100], [1,16,16], mid_lengths=[512, 256])
    out = model(test_x)
    print(out.size())
        
        