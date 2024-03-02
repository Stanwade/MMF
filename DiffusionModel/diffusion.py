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
        self.save_hyperparameters()
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
        
        self.register_buffer('alpha_bars',alpha_bars)
        
        
    def sample_forward(self, xt, t, eps=None):
        with torch.no_grad():
            alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
            if eps is None:
                eps = torch.randn_like(xt)
            else:
                eps = eps
            xt = torch.sqrt(1 - alpha_bar) * xt + torch.sqrt(alpha_bar) * eps
            return xt
    
    def training_step(self, batch, batch_idx):
        _, x = batch
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),),device=self.device)
        
        xt = self.sample_forward(x, t, eps)
        
        eps_theta = self.unet(xt, t)
        
        train_loss = F.mse_loss(eps_theta, eps)
        self.log('train_loss', train_loss, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate', lr, on_step=True, sync_dist=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        _, x = batch
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),),device=self.device)
        
        xt = self.sample_forward(x, t, eps)
        
        eps_theta = self.unet(xt, t)
        
        val_loss = F.mse_loss(eps_theta, eps)
        self.log('val_loss', val_loss, sync_dist=True)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.1,
                                                         patience=5)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                    }
                }
            
    def sample_backward_step(self, xt, t, net, simple_var=False):
        batch_size = xt.shape[0]
        t_tensor = torch.tensor([t]*batch_size, dtype=torch.long, device=xt.device).unsqueeze(1) # (n, 1)
        eps = net(xt, t_tensor)
        
        if t == 0:
            noise = torch.zeros_like(eps)
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(xt) * torch.sqrt(var)
            
        mean = (xt - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        xt = mean + noise
        
        return xt
    
    def sample_backward(self, img, net, device, simple_var=False):
        xt = img.to(device)
        net = net.to(device)
        
        for t in reversed(range(self.n_steps)):
            xt = self.sample_backward_step(xt, t, net, simple_var)
            
        return xt
    
    def sample_backward_ddim(self, img, net, device, ddim_steps=20, eta=0.0, simple_var=False):
        if simple_var:
            eta = 1
        ts = torch.linspace(self.n_steps - 1, 0, ddim_steps + 1) # size: (ddim_steps + 1)

        x = img.to(device)
        batch_size = x.shape[0]
        net = net.to(device)
        
        for i in reversed(range(1, ddim_steps + 1)):
            current_t = ts[i - 1] - 1
            prev_t = ts[i] - 1
            
            # notation ab for alpha bar
            ab_current = self.alpha_bars[current_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

            t_tensor = torch.tensor([current_t] * batch_size,
                                    dtype=torch.long).to(device).unsqueeze(1) # (n, 1)
            eps = net(x, t_tensor)
            var = eta * (1 - ab_prev) / (1 - ab_current) * (1 - ab_current / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_current)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_current) / ab_current)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_current / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term

        return x
        
if __name__ == '__main__':
    unet_config = {
        'blocks': 2,
        'img_channels': 1,
        'base_channels': 64,
        'ch_mult': [1,2,4,4],
        'norm_type': 'batchnorm',
        'activation': 'mish',
        'with_attn': True,
        'down_up_sample': True
    }
    model = DiffusionModel(unet_config)
    
    batch = [torch.randn(32,1,100,100), torch.randn(32, 1, 16, 16)]
    loss = model.training_step(batch, 0)
    
    print('loss:',loss)