import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from typing import Union, List
from torchvision.utils import make_grid
from VGGModel.networks import VGGNet, CustomVGG

class VGGModel(pl.LightningModule):
    def __init__(self,in_img_shape,label_size,input_channel):
        super().__init__()
        self.save_hyperparameters()
        self.model = CustomVGG(label_size=label_size,
                               input_channel=input_channel,
                               input_size=in_img_shape[-1],
                               channel_list_conv=[64, 128, 256], 
                               blocks=[2, 2, 3])
        self.input_size = in_img_shape

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate', lr, on_step=True, sync_dist=True)
        
        with torch.no_grad():
            if batch_idx % 100 == 0:
                num_samples = 3
                resize = transforms.Resize(size=(100,100),interpolation=transforms.InterpolationMode.NEAREST)
                
                x = x[:num_samples]
                y = y[:num_samples]
                y_hat = y_hat[:num_samples]
                
                x = resize(x) / torch.max(x)
                y = resize(y) / torch.max(y)
                y_hat = resize(y_hat) / torch.max(y_hat)
                
                combine = torch.cat((x,y,y_hat), dim=0)
                
                grid = make_grid(combine, nrow=num_samples)
                
                self.logger.experiment.add_image('img_log',grid,self.global_step)
                
        return loss
    
    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            sampleImg=torch.randn(self.input_size).unsqueeze(0).to('cuda')
            self.logger.experiment.add_graph(self.model,sampleImg)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(f'wasd: y_hat shape:{y_hat.shape}, y_shape:{y.shape}')
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }