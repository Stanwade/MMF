import matplotlib.pyplot as plt
from inspect import isfunction
from torch.utils.data import DataLoader
from datasets import MNISTDataset, MMFDataset, MMFGrayScaleDataset


def plot_imgs(inputs,name:str, dir:str='imgs', figsize = (16,16)):
    fig, axes = plt.subplots(nrows=1, ncols=inputs.size(0), figsize=figsize)
    for idx in range(inputs.size(0)):
        axes[idx].imshow(inputs[idx].squeeze().numpy(), cmap='gray')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{dir}/{name}.png')
    
def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def create_dataloader(dataset_type: str, root: str='./datasets/', target_pipeline = None):
    if dataset_type == 'MNIST':
        train_dataset = MNISTDataset(root='./datasets', train=True, transform=target_pipeline)
        validation_dataset = MNISTDataset(root='./datasets', train=False, transform=target_pipeline)
        
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=96)
        validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False, num_workers=96)
        
    elif dataset_type == 'MMF':
        # Load data
        train_dataset = MMFDataset(root='./datasets/100m_200/16x16/1',
                                train=True,
                                target_transform=target_pipeline)
        validation_dataset = MMFDataset(root='./datasets/100m_200/16x16/1',
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=96)
        validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=96)
    
    elif dataset_type =='MMFGrayscale':
        # Load data
        train_dataset = MMFGrayScaleDataset(root='./datasets',
                                            train=True,
                                            target_transform=target_pipeline)
        validation_dataset = MMFGrayScaleDataset(root='./datasets',
                                        train=False,
                                        target_transform=target_pipeline)
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=96)
        validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=96)
    
    elif dataset_type == 'MMFMNIST':
        train_dataset = MMFDataset()
    
    return train_loader, validation_loader