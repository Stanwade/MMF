import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import pytorch_lightning as pl

from .networks import UNet


class SimpleDiffusion(pl.LightningModule):
    def __init__(self, unet_config: dict):
        super().__init__()
        self.unet = UNet(**unet_config)
    
    
    def sample_step(self, xt, t, dt):
        assert dt < t
        model_out = self.unet(xt, t)
        next_x = xt + model_out * dt
        return next_x
        
    def forward(self, x0, t, dt=0.05):
        total_dt = 0
        t = torch.ones(xt.size(0))
        for i in range(int(1/dt)):
            xt = self.sample_step(xt, t, dt)
            total_dt += dt
            t = t-dt
        x0 = self.sample_step(xt, t, t)
        return x0
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        noise = torch.randn_like(x)
        t = torch.rand((x.size(0)),device=x.device)
        
        xt = (1 - t.view(-1, 1, 1, 1)) * x + t.view(-1, 1, 1, 1) * noise
        target = x - noise
        
        model_out = self.unet(xt, t)
        loss = F.mse_loss(model_out, target)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == '__main__':
    unet_config = {
        'blocks': 2,
        'img_channels': 3,
        'base_channels': 64,
        'ch_mult': [1,2,4,4],
        'norm_type': 'batchnorm',
        'activation': 'lrelu',
        'with_attn': True,
        'mid_attn': True,
        'down_up_sample': True
    }
    model = SimpleDiffusion(unet_config=unet_config)
    
    batch = [torch.randn(3, 3, 64, 64),torch.rand(4)]
    loss = model.training_step(batch, 0)
    
    print(loss)