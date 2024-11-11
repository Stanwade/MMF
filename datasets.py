from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from torchvision import io
from torchvision.io import read_image
from torchvision.datasets import MNIST, ImageFolder
from typing import Literal
import glob

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
        
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder,'speckles_1plane.npy')))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, 'pattern_gray.npy')))
        
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
        target = self.target[idx]
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float() / 255
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target

class MMF_FMNISTDataset_grayscale(Dataset):
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
        self.data_folder = os.path.join(root, 'FMNIST_grayscale')
        
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder,'speckles_1plane.npy')))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, 'pattern_gray.npy')))
        
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
        target = self.target[idx]
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float() / 255
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target
    
class MMF_imgNet32Dataset(Dataset):
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
        self.data_folder = os.path.join(root, 'imgnet32_rgbcycle_1')
        
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder,'speckles_downsampled.npy')))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, 'pattern_gray.npy')))
        
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
        target = self.target[idx]
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float() / 255
        target = target.reshape(-1,32,32)
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target
    
class MMF_imgNet64Dataset(Dataset):
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
        self.data_folder = os.path.join(root, 'img64')
        
        self.data = torch.from_numpy((np.load(os.path.join(self.data_folder,'speckles_intensity_only.npy'))).astype(np.float32))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, '64_pattern_sum.npy')).astype(np.float32))
        
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
        target = self.target[idx]
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float() / 255
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target

class MMF_imgNet16Dataset(Dataset):
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
        self.data_folder = os.path.join(root, 'imgnet16')
        
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder,'speckles_merged.npy')).astype(np.float32))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, 'pattern_gray.npy')).astype(np.float32))
        
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
        target = self.target[idx]
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float() / 255
        target = target.reshape(-1,16,16)
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target

class MMF_SingleImageDataset(Dataset):
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
        self.data_folder = os.path.join(root, 'leopard_2k')
        
        self.data = torch.from_numpy(np.load(os.path.join(self.data_folder,'speckles_merged_fixed.npy')).astype(np.float32))
        self.target = torch.from_numpy(np.load(os.path.join(self.data_folder, 'pattern_gray.npy')).astype(np.float32))
        
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
        target = self.target[idx]
        
        img = img.unsqueeze(0).float()
        target = target.unsqueeze(0).float() / 255
        target = target.reshape(-1,32,32)
        
        if self.target_transform:
            target = self.target_transform(target)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target

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
        img = read_image(filepath, mode=io.ImageReadMode.RGB).float() / 255
        # crop to expected size, do random resize and rotation first
        if self.expected_size:
            img = transforms.RandomResizedCrop(self.expected_size,
                                               scale=(0.8,1),
                                               ratio=(9/10,10/9),
                                               interpolation=transforms.InterpolationMode.BICUBIC)(img)
        if self.transforms_pipeline:
            img = self.transforms_pipeline(img)
        
        # normalize
        img = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))(img)
        
        return self.labels[idx], img

def create_dataloader(dataset_type: Literal['MNIST', 
                                            'MMF', 
                                            'MMFGrayscale', 
                                            'MMFMNIST', 
                                            'MMFMNIST_GRAY', 
                                            'MMFFMNIST_GRAY',
                                            'imgnet16', 
                                            'imgnet32', 
                                            'imgnet64', 
                                            'leopard2k',
                                            'image_folder'],
                      root: str='./datasets/',
                      target_pipeline = None,
                      batch_size: int = 64,
                      num_workers: int = 96,
                      need_datasets: bool = False):
    assert dataset_type in ('MNIST', 
                            'MMF', 
                            'MMFGrayscale', 
                            'MMFMNIST', 
                            'MMFMNIST_GRAY', 
                            'MMFFMNIST_GRAY', 
                            'imgnet16',
                            'imgnet32', 
                            'imgnet64', 
                            'leopard2k'), f'dataset_type invalid, got {dataset_type}'
    if dataset_type == 'MNIST':
        train_dataset = MNISTDataset(root=root, train=True, transform=target_pipeline)
        validation_dataset = MNISTDataset(root=root, train=False, transform=target_pipeline)
        
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    elif dataset_type == 'MMF':
        # Load data
        train_dataset = MMFDataset(root=root,
                                train=True,
                                target_transform=target_pipeline)
        validation_dataset = MMFDataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif dataset_type =='MMFGrayscale':
        # Load data
        train_dataset = MMFGrayScaleDataset(root=root,
                                            train=True,
                                            target_transform=target_pipeline)
        validation_dataset = MMFGrayScaleDataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif dataset_type == 'MMFMNIST':
        train_dataset = MMFMNISTDataset(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMFMNISTDataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == 'MMFMNIST_GRAY':
        train_dataset = MMFMNISTDataset_grayscale(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMFMNISTDataset_grayscale(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif dataset_type == 'MMFFMNIST_GRAY':
        train_dataset = MMF_FMNISTDataset_grayscale(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMF_FMNISTDataset_grayscale(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == 'imgnet16':
        train_dataset = MMF_imgNet16Dataset(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMF_imgNet16Dataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == 'imgnet32':
        train_dataset = MMF_imgNet32Dataset(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMF_imgNet32Dataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == 'imgnet64':
        train_dataset = MMF_imgNet64Dataset(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMF_imgNet64Dataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == 'leopard2k':
        train_dataset = MMF_SingleImageDataset(root=root,
                                        train=True,
                                        target_transform=target_pipeline)
        validation_dataset = MMF_SingleImageDataset(root=root,
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif dataset_type == 'image_folder':
        train_dataset = imgFolderDataset(root=root,
                                         expected_size=(64,64),
                                         postfix='.jpeg',
                                         target_transform=target_pipeline)
        validation_dataset = imgFolderDataset(root=root,
                                         expected_size=(64,64),
                                         postfix='.jpeg',
                                         target_transform=target_pipeline)
    else:
        raise NotImplementedError(f"dataset type {dataset_type} doesn't exist!")
    
    if need_datasets:
        return train_dataset, validation_dataset, train_loader, validation_loader
    else:
        return train_loader, validation_loader

if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    
    img_size = 48
    print(torchvision.__version__)
    
    target_pipeline = transforms.Compose([
        
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])
    
    dataset = MMFMNISTDataset(root='./datasets', train=True)
    
    a = dataset.__getitem__(0)
    print(a[0].shape)
    print(a[1].shape)
    
    print(a[0].dtype)
    print(a[1].dtype)
    
    print(f'data max {torch.max(a[0])}')
    print(f'data min {torch.min(a[0])}')
    
    print(f'label max {torch.max(a[1])}')
    print(f'label min {torch.min(a[1])}')
    
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    test_data = dataloader.__iter__().__next__()
    
    
    from utils import plot_imgs
    plot_imgs(test_data[0], 'test1')
    plot_imgs(test_data[1], 'test2')