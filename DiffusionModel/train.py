from DiffusionModel.diffusion import SimpleDiffusion

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import MMFDataset

# train.py

def train_diffusion_model(diffusion_model, train_loader, validation_loader, num_epochs=100, callbacks=[]):
    trainer = Trainer(max_epochs=num_epochs)
    diffusion_model_trainer = diffusion_model
    trainer.fit(diffusion_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader,
                callbacks=callbacks)


if __name__ == '__main__':
    # set config
    pl.seed_everything(0)
    
    # set unet configs
    unet_config = {
        'blocks': 2,
        'img_channels': 1,
        'base_channels': 4,
        'ch_mult': [1,2,4,4],
        'norm_type': 'batchnorm',
        'activation': 'lrelu',
        'with_attn': True,
        'mid_attn': True,
        'down_up_sample': True
    }
    
    # set modelcheckpoint config
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.2f}',
        mode='min',
        every_n_epochs=10,
        save_top_k=3,
        save_last=True
    )
    
    # create model
    model = SimpleDiffusion(unet_config=unet_config)
    
    # Load data
    train_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=True)
    validation_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=False)
    # create loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    
    train_diffusion_model(model, train_loader, validation_loader, callbacks=[model_checkpoint])