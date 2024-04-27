from UNetModel.model import UNetModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import create_dataloader

from typing import Optional

def train_unet_model(reconstruction_model,
                               train_loader,
                               validation_loader,
                               logger,
                               num_epochs=100,
                               callbacks=[]):
    
    trainer = Trainer(max_epochs=num_epochs,
                      callbacks=callbacks,
                      gradient_clip_val=0.6,
                      logger=logger,
                      devices='0,1,4,5')
    reconstruction_model_trainer = reconstruction_model
    trainer.fit(reconstruction_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)
    
if __name__ == '__main__':
    
    pl.seed_everything(42)
    logger = TensorBoardLogger(save_dir='./',log_graph=True)
    
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        dirpath='UNetModel/ckpts_unet_test',
        mode='min',
        every_n_epochs=10,
        save_top_k=3,
        save_last=True
    )
    
    dataset_type = 'MMFMNIST'
    label_size = 32
    
    target_pipeline = transforms.Compose(
        [transforms.Resize((200,200),interpolation=transforms.InterpolationMode.NEAREST)]
    )
    
    train_dataset, valid_dataset, train_loader, valid_loader = create_dataloader(dataset_type,need_datasets=True, target_pipeline=target_pipeline)
    
    
    reconstruction_model = UNetModel(logger=logger,
                                     in_img_shape=train_dataset[0][0].shape,
                                     base_channels=16,
                                     ch_mult=[1, 2, 4, 4],
                                     norm_type='batchnorm',
                                     activation='lrelu',
                                     with_attn=[False, False, True, True],
                                     down_up_sample=True)
                                     
    
    train_unet_model(reconstruction_model,
                               train_loader,
                               valid_loader,
                               logger=logger,
                               num_epochs = 100,
                               callbacks = [model_checkpoint_callback])