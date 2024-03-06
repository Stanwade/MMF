import matplotlib.pyplot as plt
from inspect import isfunction
from torch.utils.data import DataLoader
from datasets import MNISTDataset, MMFDataset, MMFGrayScaleDataset, MMFMNISTDataset


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

def create_dataloader(dataset_type: str,
                      root: str='./datasets/',
                      target_pipeline = None,
                      batch_size: int = 64,
                      num_workers: int = 96,
                      need_datasets: bool = False):
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
        
    else:
        raise NotImplementedError(f"dataset type {dataset_type} doesn't exist!")
    
    if need_datasets:
        return train_dataset, validation_dataset, train_loader, validation_loader
    else:
        return train_loader, validation_loader