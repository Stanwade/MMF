from ReconstructionModel.model import ReconstructionModel

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import create_dataloader

def train_reconstruction_model(reconstruction_model,
                               train_loader,
                               validation_loader,
                               num_epochs=200,
                               callbacks=[]):
    
    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, gradient_clip_val=0.6)
    reconstruction_model_trainer = reconstruction_model
    trainer.fit(reconstruction_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)
    
if __name__ == '__main__':
    
    pl.seed_everything(0)
    
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        dirpath='ReconstructionModel/ckpts_mnist',
        mode='min',
        every_n_epochs=10,
        save_top_k=3,
        save_last=True
    )
    
    dataset_type = 'MMFMNIST'
    label_size = 32
    
    train_dataset, valid_dataset, train_loader, valid_loader = create_dataloader(dataset_type,need_datasets=True)
    
    reconstruction_model = ReconstructionModel(in_img_shape=train_dataset[0][0].shape,
                                               out_img_shape=train_dataset[0][1].shape,
                                               mid_lengths=[1024, 512],
                                               norm_type='batchnorm',
                                               img_size=label_size,
                                               activation='none')
    
    train_reconstruction_model(reconstruction_model,
                               train_loader,
                               valid_loader,
                               num_epochs = 200,
                               callbacks = [model_checkpoint_callback])