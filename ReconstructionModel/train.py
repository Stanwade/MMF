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
from datasets import MMFDataset

def train_reconstruction_model(reconstruction_model,
                               train_loader,
                               validation_loader,
                               num_epochs=100,
                               callbacks=[]):
    
    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks)
    reconstruction_model_trainer = reconstruction_model
    trainer.fit(reconstruction_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)
    
if __name__ == '__main__':
    train_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=True)
    valid_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=False)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Valid dataset size: {len(valid_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    print(f'Train loader size: {len(train_loader)}')
    print(f'Valid loader size: {len(valid_loader)}')
    
    reconstruction_model = ReconstructionModel(in_img_shape=train_dataset[0][0].shape,
                                               out_img_shape=train_dataset[0][1].shape)
    
    train_reconstruction_model(reconstruction_model, train_loader, valid_loader)