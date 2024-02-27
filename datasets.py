from torch.utils.data import Dataset
import torch
import numpy as np
import os
from torchvision import transforms

class MMF100m200Dataset(Dataset):
    
    def __init__(self,root='./datasets/100m_200',resolution=(16,16)) -> None:
        
        super().__init__()
        
        if resolution == (16,16):
            self.root = root + '/16x16/1/'
            self.imagedata = torch.from_numpy(np.load(self.root + 'pattern.npy'))
            self.labeldata = torch.from_numpy(np.load(self.root + 'SI_200_0.22_100m.npy'))
        
        elif resolution == (32,32):
            self.root = root + '/32x32/2/'
            self.imagedata = torch.from_numpy(np.load(self.root + 'pattern.npy'))
            self.labeldata = torch.from_numpy(np.load(self.root + 'SI_100_0.22_100m_100x100.npy'))
        
        else:
            exit('MMF100m200: resolution does not exist!')
    
        self.filenames = sorted(os.listdir(self.root))
        
        
    def __len__(self) -> int:
        return len(self.imagedata), len(self.labeldata)
    
    def __getitem__(self, index):
        image = self.imagedata[index].reshape(-1,16)
        label = self.labeldata[index]
        return image, label

class MMFDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, train_size=0.7, valid_size=0.15):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # Load the dataset
        self.data = torch.from_numpy(np.load(os.path.join(self.root, 'SI_200_0.22_100m.npy')))
        self.targets = torch.from_numpy(np.load(os.path.join(self.root, 'pattern.npy')))

        # Split dataset into train, validation, and test sets
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split_train = int(np.floor(train_size * dataset_size))
        split_valid = int(np.floor((train_size + valid_size) * dataset_size))

        if train:
            self.data = self.data[:split_train]
            self.targets = self.targets[:split_train]
        else:
            valid_data = self.data[split_train:split_valid]
            valid_targets = self.targets[split_train:split_valid]
            test_data = self.data[split_valid:]
            test_targets = self.targets[split_valid:]
            self.data, self.targets = (valid_data, valid_targets) if len(valid_data) > len(test_data) else (test_data, test_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx] / 255
        target = self.targets[idx].reshape(-1, 16)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        
        return image.unsqueeze(0), target.float().unsqueeze(0)

if __name__ == '__main__':
    dataset = MMFDataset(root='./datasets/100m_200/16x16/1/')
    a = dataset.__getitem__(0)
    exit('finished debug!')