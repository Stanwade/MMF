from torch.utils.data import Dataset
import torch
import numpy as np
import os
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

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
        self.root = os.path.join(root,'100m_200','16x16','1')
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
        
        image = image.unsqueeze(0)
        target = target.unsqueeze(0).float()
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        
        return image, target


class MNISTDataset(Dataset):
    """MNIST Dataset."""

    def __init__(self, root='./datasets', train=True, transform=None, download=False):
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        self.train = train  # training set or test set
        self.transform = transform

        self.data_folder = os.path.join(root, 'MNIST', 'raw')

        # download the data
        if download:
            self.data = MNIST(root, train=self.train, transform=transform, download=True)

        # load the data
        else:
            self.data = MNIST(root, train=self.train, transform=transform, download=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = self.data[idx][1]
        pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        return label, pipeline(img)


class MMFGrayScaleDataset(Dataset):
    def __init__(self, root='./datasets',
                 train=True,
                 transform=None,
                 target_transform = None,
                 train_size=0.7,
                 valid_size=0.15
                 ) -> None:
        super().__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_folder = os.path.join(root, 'MMF_grayscale')
        
        # Load the dataset
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder, 'speckles.npy')))
        self.targets = torch.from_numpy(np.load(os.path.join(self.data_folder, 'pattern.npy')))
        
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
        img = self.data[idx] / 255
        target = self.targets[idx].reshape(-1, 16).float() / 3
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float()
        
        if self.target_transform:
            target = self.target_transform(target)
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

class MMFMNISTDataset(Dataset):
    def __init__(self,
                 root='./datasets/',
                 train=True,
                 transform=None,
                 target_transform = None,
                 train_size=0.7,
                 valid_size=0.15) -> None:
        super().__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_folder = os.path.join(root, 'MMF_MNIST')
        
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder,'mnist.npy')))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, 'pattern.npy')))
        
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split_train = int(np.floor(train_size * dataset_size))
        split_valid = int(np.floor((train_size + valid_size) * dataset_size))
        
        if train:
            self.data = self.data[:split_train]
            self.target = self.target[:split_train]
        else:
            valid_data = self.data[split_train:split_valid]
            valid_targets = self.target[split_train:split_valid]
            test_data = self.data[split_valid:]
            test_targets = self.target[split_valid:]
            self.data, self.target = (valid_data, valid_targets) if len(valid_data) > len(test_data) else (test_data, test_targets)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.target[idx].reshape(-1, 32)
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float()
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target
    
class MMFMNISTDataset_grayscale(Dataset):
    def __init__(self,
                 root='./datasets/',
                 train=True,
                 transform=None,
                 target_transform = None,
                 train_size=0.7,
                 valid_size=0.15) -> None:
        super().__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_folder = os.path.join(root, 'MNIST_grayscale')
        
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder,'pattern_gray.npy')))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, 'speckles_1plane.npy')))
        
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split_train = int(np.floor(train_size * dataset_size))
        split_valid = int(np.floor((train_size + valid_size) * dataset_size))
        
        if train:
            self.data = self.data[:split_train]
            self.target = self.target[:split_train]
        else:
            valid_data = self.data[split_train:split_valid]
            valid_targets = self.target[split_train:split_valid]
            test_data = self.data[split_valid:]
            test_targets = self.target[split_valid:]
            self.data, self.target = (valid_data, valid_targets) if len(valid_data) > len(test_data) else (test_data, test_targets)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.target[idx].reshape(-1, 32)
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float()
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target

if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    
    img_size = 48
    print(torchvision.__version__)
    
    target_pipeline = transforms.Compose([
        
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])
    
    dataset = MMFGrayScaleDataset(root='./datasets', train=True)
    
    a = dataset.__getitem__(0)
    print(a[0].shape)
    print(a[1].shape)
    
    print(a[0].dtype)
    print(a[1].dtype)
    
    print(f'label max {torch.max(a[1])}')
    print(f'label min {torch.min(a[1])}')
    
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    test_data = dataloader.__iter__().__next__()
    
    
    from utils import plot_imgs
    plot_imgs(test_data[0], 'test1')
    plot_imgs(test_data[1], 'test2')