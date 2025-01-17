from DiffusionModel.diffusion import DiffusionModel, SimpleDiffusion

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

# train.py

def train_diffusion_model(diffusion_model, train_loader, validation_loader, num_epochs=100, callbacks=[]):
    torch.set_float32_matmul_precision('high')
    trainer = Trainer(max_epochs=num_epochs, 
                      callbacks=callbacks, 
                      devices='-1')
    diffusion_model_trainer = diffusion_model
    trainer.fit(diffusion_model_trainer,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader,
                ckpt_path='./DiffusionModel/ckpts_celebHQ64_wo_condition/last.ckpt')


if __name__ == '__main__':
    # set config
    pl.seed_everything(43)
    torch.cuda.empty_cache()
    
    # set unet configs
    unet_config = {
        'blocks': 2,
        'img_channels': 3,
        'base_channels': 64,
        'ch_mult': [1,2,4,8,8],
        'norm_type': 'groupnorm',
        'activation': 'mish',
        'pe_dim': 1024,
        'with_attn': [False, True, True, True, False],
        'down_up_sample': True,
        'condition_channels': 1
    }
    
    # set modelcheckpoint config
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{val_loss:.4f}',
        dirpath='DiffusionModel/ckpts_celebHQ64_wo_condition',
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
    
    img_size = 64
    
    dataset_type = 'celebHQ64'
    
    # create model
    model = DiffusionModel(unet_config=unet_config, 
                           cfg=None, 
                           reconstruction_model_dir=None)
    
    # define target transform pipeline, turn a [1,16,16] into [1,64,64]
    target_pipeline = transforms.Compose([
        transforms.Resize( (img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    ])
    
    train_loader, validation_loader = create_dataloader(dataset_type=dataset_type, target_pipeline=target_pipeline)
    # from datasets import imgFolderDataset
    # train_set = imgFolderDataset('./datasets/ILSVRC2012_img_val',expected_size=(64,64),postfix='.JPEG')
    # train_loader = DataLoader(train_set,batch_size=128,shuffle=True)
    # indices = torch.randperm(1000)[:512]
    # valid_sampler = SubsetRandomSampler(indices)
    # validation_loader = DataLoader(train_set, batch_size=128,shuffle=False,sampler=valid_sampler)
    
    train_diffusion_model(model,
                          train_loader,
                          validation_loader,
                          num_epochs=1200,
                          callbacks=[model_checkpoint_callback])