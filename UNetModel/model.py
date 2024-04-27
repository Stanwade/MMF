import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from typing import Union, List

from UNetModel.networks import UNet
from torchvision.utils import make_grid

class UNetModel(pl.LightningModule):
    def __init__(self, 
                 logger: TensorBoardLogger,
                 in_img_shape: torch.Tensor,
                 base_channels: int,
                 ch_mult: list,
                 norm_type: str,
                 activation: str,
                 with_attn: Union[bool, List[bool]],
                 down_up_sample: bool = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            blocks=2,
            img_channels=in_img_shape[0],
            base_channels=base_channels,
            ch_mult=ch_mult,
            norm_type=norm_type,
            activation=activation,
            with_attn=with_attn,
            down_up_sample=down_up_sample
        )
        self.input_size = in_img_shape
        # self.logger:TensorBoardLogger = logger

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
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)
        
        # outputs = {'input' : x,
        #            'output' : y_hat,
        #            'label' : y,
        #            'loss' : loss}
        return loss
    
    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            sampleImg=torch.randn(self.input_size).unsqueeze(0).to('cuda')
            self.logger.experiment.add_graph(self.model,sampleImg)
    
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