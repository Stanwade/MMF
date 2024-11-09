import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import os
import numpy as np

dir = './datasets/imgnet32_rgbcycle_1'
filename = 'pattern_gray.npy'
new_filename = 'pattern_gray_16x16.npy'
origin_patterns = torch.from_numpy(np.load(os.path.join(dir, filename))).reshape(-1,1,32,32)
print(f'origin patterns shape: {origin_patterns.shape}')

ds_size = 16
downsample = transforms.Compose([
    transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.NEAREST)
])

print('down sampling...')
new_patterns = downsample(origin_patterns).reshape(origin_patterns.shape[0],-1)
print(f'new patterns shape: {new_patterns.shape}')
# exit('debug')
# turn torch tensor to numpy
new_patterns = new_patterns.numpy()

np.save(os.path.join(dir, new_filename), new_patterns)

new_patterns = np.load(os.path.join(dir,new_filename))
print(f'eval: new_patterns shape: {new_patterns.shape}')