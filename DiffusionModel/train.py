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
    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, devices='0,1')
    diffusion_model_trainer = diffusion_model
    trainer.fit(diffusion_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)


if __name__ == '__main__':
    # set config
    pl.seed_everything(42)
    torch.cuda.empty_cache()
    
    # set unet configs
    unet_config = {
        'blocks': 2,
        'img_channels': 3,
        'base_channels': 10,
        'ch_mult': [1,2,4,8,8],
        'norm_type': 'batchnorm',
        'activation': 'relu',
        'pe_dim': 128,
        'with_attn': [False, False, False, False, True],
        'down_up_sample': False
    }
    
    # set modelcheckpoint config
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        dirpath='DiffusionModel/ckpts_imgnet32_cfg_wo_Down_Up',
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
    
    dataset_type = 'imgnet32'
    
    # create model
    model = DiffusionModel(unet_config=unet_config, 
                           cfg=3.0, 
                           reconstruction_model_dir='./ReconstructionModel/ckpts_imgnet32_real/epoch=99-val_loss=0.0078-v1.ckpt')
    
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