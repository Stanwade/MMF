import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import os
import numpy as np

dir = ''
filename = 'patterns_gray.npy'
new_filename = 'patterns_gray_16x16.npy'
origin_patterns = torch.from_numpy(np.load(os.path.join(dir, filename)))

ds_size = 16
downsample = transforms.Compose([
    transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.NEAREST)
])

new_patterns = downsample(origin_patterns)

torch.save(new_patterns, os.path.join(dir, new_filename))