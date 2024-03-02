import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ReconstructionModel.networks import FullyConnectedNetwork

class ReconstructionModel(pl.LightningModule):
    def __init__(self, in_img_shape: torch.Tensor, out_img_shape: torch.Tensor, **kwargs):
        super().__init__()
        self.model = FullyConnectedNetwork(in_img_shape, out_img_shape, **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)
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