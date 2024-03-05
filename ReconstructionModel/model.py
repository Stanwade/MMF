import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ReconstructionModel.networks import FullyConnectedNetwork

class ReconstructionModel(pl.LightningModule):
    def __init__(self, in_img_shape: torch.Tensor,
                 out_img_shape: torch.Tensor,
                 mid_lengths: list,
                 norm_type:str,
                 activation:str,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = FullyConnectedNetwork(in_img_shape,
                                           out_img_shape,
                                           mid_lengths=mid_lengths,
                                           norm_type=norm_type,
                                           activation=activation,
                                           **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate', lr, on_step=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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