from DiffusionModel.diffusion import DiffusionModel, SimpleDiffusion
from DiffusionModel.control_diffusion import ControlDiffusionModel

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from datasets import create_dataloader

def train_controlnet(controled_diffusion_model, 
                     train_loader, 
                     validation_loader, 
                     num_epochs=100, 
                     ckpt_root_path='./DiffusionModel/ControlNet_ckpts/',
                     callbacks=[]):
    torch.set_float32_matmul_precision('high')
    trainer = Trainer(max_epochs=num_epochs, 
                      callbacks=callbacks, 
                      devices='-1')
    controled_diffusion_model_trainer = controled_diffusion_model
    trainer.fit(controled_diffusion_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader,
                ckpt_path=ckpt_root_path + 'last.ckpt') # train from last checkpoint

if __name__ == "__main__":
    # set config
    pl.seed_everything(43)
    torch.cuda.empty_cache()
    
    diffusion_dir = './DiffusionModel/ckpts_celebHQ64_wo_condition/last.ckpt'
    
    # set modelcheckpoint config
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        dirpath='./DiffusionModel/ControlNet_ckpts',
        mode='min',
        every_n_epochs=10,
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        mode = 'min'
    )
    
    # create dataloader
    train_loader, validation_loader = create_dataloader(batch_size=1, image_size=64, dataset='celebA')
    
    # create model
    controled_diffusion_model = ControlDiffusionModel(diffusion_model_dir=diffusion_dir,
                                                      hint_channels=3)
    
    # train model
    train_controlnet(controled_diffusion_model, 
                     train_loader, 
                     validation_loader, 
                     num_epochs=100, 
                     ckpt_root_path='./DiffusionModel/ControlNet_ckpts/',
                     callbacks=[model_checkpoint_callback, early_stop_callback])