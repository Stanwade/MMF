import torch
from torchvision import datasets, transforms, io
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import glob
import os
from matplotlib import pyplot as plt

class imgFolderDataset(Dataset):
    
    def __init__(self, root, expected_size=None, transforms_pipeline=None, postfix='.png'):
        super().__init__()
        self.filenames = glob.glob(root + '/*' + postfix)
        # take filename as label
        self.labels = [f.split('\\')[-1].split('.')[0] for f in self.filenames]
        self.expected_size = expected_size
        self.transforms_pipeline = transforms_pipeline

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = self.filenames[idx]
        img = read_image(filepath, mode=io.ImageReadMode.RGB)
        # crop to expected size, do random resize and rotation first
        if self.expected_size:
            img = transforms.RandomResizedCrop(self.expected_size,
                                               interpolation=transforms.InterpolationMode.BICUBIC)(img)
        if self.transforms_pipeline:
            img = self.transforms_pipeline(img)
        return img, self.labels[idx]
    
if "__main__" == __name__:
    root = './imgs'
    import numpy as np
    a = np.load('./datasets/img64/64_pattern_sum.npy')
    print(a.shape)
    b = (np.load('./datasets/img64/speckles_intensity_only.npy'))
    print(b.shape)