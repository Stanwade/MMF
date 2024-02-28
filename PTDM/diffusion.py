import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import pytorch_lightning as pl

from DiffusionModel.networks import UNet


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
        for _ in range(int(1/dt)):
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
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        noise = torch.randn_like(x)
        t = torch.rand((x.size(0)),device=x.device)
        
        xt = (1 - t.view(-1, 1, 1, 1)) * x + t.view(-1, 1, 1, 1) * noise
        target = x - noise
        
        model_out = self.unet(xt, t)
        val_loss = F.mse_loss(model_out, target)
        self.log('val_loss', val_loss, sync_dist=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


class DiffusionModel(pl.LightningModule):
    def __init__(self, unet_config: dict,
                 n_steps: int = 1000,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__()
        self.unet = UNet(**unet_config)
        self.n_steps = n_steps
        self.min_beta = min_beta
        self.max_beta = max_beta
        
        betas = torch.linspace(min_beta, max_beta, n_steps)
        self.betas = betas
        
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        
        for i in range(n_steps):
            product *= alphas[i]
            alpha_bars[i] = product
        
        self.alpha_bars = alpha_bars
        
    def sample_forward_step(self, xt, t, eps=None):
        self.alpha_bars = self.alpha_bars.to(device=xt.device)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(xt)
        else:
            eps = eps
        xt = torch.sqrt(1 - alpha_bar) * xt + torch.sqrt(alpha_bar) * eps
        return xt
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),),device=x.device)
        
        xt = self.sample_forward_step(x, t, eps)
        
        eps_theta = self.unet(xt, t)
        
        train_loss = F.mse_loss(eps_theta, eps)
        # self.log('train_loss', train_loss, sync_dist=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),),device=x.device)
        
        xt = self.sample_forward_step(x, t, eps)
        
        eps_theta = self.unet(xt, t)
        
        val_loss = F.mse_loss(eps_theta, eps)
        self.log('val_loss', val_loss, sync_dist=True)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
            
            
            
class ddpm():
    def __init__(self,unet_config: dict,
                 n_steps: int = 1000,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        self.unet = UNet(**unet_config)
        self.n_steps = n_steps
        self.min_beta = min_beta
        self.max_beta = max_beta
        
        betas = torch.linspace(min_beta, max_beta, n_steps)
        self.betas = betas
        
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        
        for i in range(n_steps):
            product *= alphas[i]
            alpha_bars[i] = product
        
        self.alpha_bars = alpha_bars
        
    def sample_forward_step(self, xt, t, eps=None):
        self.alpha_bars = self.alpha_bars.to(device=xt.device)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(xt)
        else:
            eps = eps
        xt = torch.sqrt(1 - alpha_bar) * xt + torch.sqrt(alpha_bar) * eps
        
        return xt
    
    
    
        

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from datasets import MMFDataset
    import time
    unet_config = {
        'blocks': 2,
        'img_channels': 1,
        'base_channels': 64,
        'ch_mult': [1,2,4,4],
        'norm_type': 'batchnorm',
        'activation': 'lrelu',
        'with_attn': True,
        'mid_attn': True,
        'down_up_sample': True
    }
    myddpm = ddpm(unet_config,1000,0.0001,0.02)
    model = myddpm.unet
    
    epochs = 100
    
    # Load data
    train_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=True)
    validation_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=False)
    # create loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)     
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # train loop
    for i in range(epochs):
        tick = time.time()
        total_loss = 0
        for batch in train_loader:
            x, _ = batch
            batchsize = x.size(0)
            eps = torch.randn_like(x)
            t = torch.randint(0, myddpm.n_steps, (x.size(0),),device=x.device)
            xt = myddpm.sample_forward_step(x, t, eps)
            eps_theta = myddpm.unet(xt, t)
            train_loss = F.mse_loss(eps_theta, eps)
            
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += train_loss.item() * batchsize
        total_loss /= len(train_loader.dataset)
        toc = time.time()
        print(f'epoch {i}, loss: {total_loss}, time: {toc-tick}')
            # self.log('train_loss', train_loss, sync_dist=True)
            