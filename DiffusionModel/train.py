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
from datasets import create_dataloader

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
        'base_channels': 10,
        'ch_mult': [1,2,4,8,8],
        'norm_type': 'batchnorm',
        'activation': 'mish',
        'pe_dim': 128,
        'with_attn': [True, True, False, False, False],
        'down_up_sample': False
    }
    
    # set modelcheckpoint config
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        dirpath='DiffusionModel/ckpts_mnist',
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
    
    dataset_type = 'MMFMNIST'
    
    # create model
    model = DiffusionModel(unet_config=unet_config)
    
    # define target transform pipeline, turn a [1,16,16] into [1,64,64]
    target_pipeline = transforms.Compose([
        transforms.Resize( (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])
    
    train_loader, validation_loader = create_dataloader(dataset_type=dataset_type, target_pipeline=target_pipeline)
    
    train_diffusion_model(model,
                          train_loader,
                          validation_loader,
                          num_epochs=100,
                          callbacks=[model_checkpoint_callback])