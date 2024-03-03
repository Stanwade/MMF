from DiffusionModel.diffusion import DiffusionModel, SimpleDiffusion

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import MMFDataset, MNISTDataset

# train.py

def train_diffusion_model(diffusion_model, train_loader, validation_loader, num_epochs=100, callbacks=[]):
    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks)
    diffusion_model_trainer = diffusion_model
    trainer.fit(diffusion_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)


if __name__ == '__main__':
    # set config
    pl.seed_everything(0)
    torch.cuda.empty_cache()
    
    # set unet configs
    unet_config = {
        'blocks': 2,
        'img_channels': 1,
        'base_channels': 32,
        'ch_mult': [1,2,4,8,8],
        'norm_type': 'batchnorm',
        'activation': 'mish',
        'with_attn': [True, True, True, False, False],
        'down_up_sample': False
    }
    
    # set modelcheckpoint config
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        dirpath='DiffusionModel/ckpts3',
        mode='min',
        every_n_epochs=10,
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.00,
        patience = 3,
        verbose = False,
        mode= 'min'
    )
    
    img_size = 32
    
    dataset_type = 'MNIST'
    
    # create model
    model = DiffusionModel(unet_config=unet_config)
    
    # define target transform pipeline, turn a [1,16,16] into [1,64,64]
    target_pipeline = transforms.Compose([
        transforms.Resize( (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])
    
    if dataset_type == 'MNIST':
        train_dataset = MNISTDataset(root='./datasets', train=True, transform=target_pipeline)
        validation_dataset = MNISTDataset(root='./datasets', train=False, transform=target_pipeline)
        
        # create loader
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=96)
        validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False, num_workers=96)
        
    else:
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
        
    train_diffusion_model(model,
                          train_loader,
                          validation_loader,
                          num_epochs=100,
                          callbacks=[model_checkpoint_callback])