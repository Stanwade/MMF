import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import math
from typing import Optional, Union, List
if __name__ == "__main__":
    # add sys path
    import sys
    sys.path.append('..')
    from ReconstructionModel.model import ReconstructionModel
else:
    from ReconstructionModel.model import ReconstructionModel


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
                 max_beta: float = 0.02,
                 cfg: Optional[float] = None,
                 cfg_drop: float = 0.25,
                 reconstruction_model_dir: str = 'ReconstructionModel/ckpts_mnist/epoch=109-val_loss=0.0176.ckpt'):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(64,1,64,64)
        self.unet = UNet(**unet_config, n_steps=n_steps, with_cond=(cfg is not None))
        self.n_steps = n_steps
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.cfg = cfg
        self.cfg_drop = cfg_drop
        
        if cfg is not None:
            # load reconstruction model
            self.r_model = ReconstructionModel.load_from_checkpoint(reconstruction_model_dir,map_location=self.device)

            # print(self.r_model)
            self.r_model.freeze()
            self.r_model.requires_grad_(False)
            # print(f'self recons model loaded, type: {type(self.r_model)}')
        
        betas = torch.linspace(min_beta, max_beta, n_steps)
        self.betas = betas
        
        alphas = 1 - betas
        self.alphas = alphas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        
        for i in range(n_steps):
            product *= alphas[i]
            alpha_bars[i] = product
        
        self.register_buffer('alpha_bars',alpha_bars)
    
    @torch.no_grad()
    def forward(self, x):
        # wrote this only for debugging
        # print(f'in forward')
        t = torch.randint(0, self.n_steps, (x.size(0),),device=self.device)
        c = torch.randint(0,255,(x.size(0),3,64,64),device=self.device)
        return self.unet(x, t, c)
    
    def sample_forward(self, xt, t, eps=None):
        self.eval()
        with torch.no_grad():
            alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
            if eps is None:
                eps = torch.randn_like(xt)
            else:
                eps = eps
            xt = torch.sqrt(1 - alpha_bar) * eps + torch.sqrt(alpha_bar) * xt
            return xt
    
    def training_step(self, batch, batch_idx):
        
        # exit('debug finished')
        _, x = batch
        
        if self.cfg is not None:
            with torch.no_grad():
                # init condition mask
                y = batch[0]
                cond_mask = torch.tensor(torch.rand(size=(len(x),)) <= self.cfg_drop, device=x.device)
                
                # make conditions
                c = self.r_model(y)
                
                # set condition mask, if True, set to 0
                c[cond_mask] = torch.ones_like(c[0])
                
                # repeat this 3 times in channel for CLIP embedding
                c = c.repeat(1,3,1,1)
                
                transform = transforms.Resize(size=(x.shape[2],x.shape[3]),
                                              interpolation=transforms.InterpolationMode.BICUBIC)
                c = transform(c)
                c = torch.clamp(c, 0, 1)
        
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),),device=self.device)
        
        xt = self.sample_forward(x, t, eps)
        
        eps_theta = self.unet(xt, t, c if self.cfg is not None else None)
        
        train_loss = F.mse_loss(eps_theta, eps)
        self.log('train_loss', train_loss, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate', lr, on_step=True, sync_dist=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        _, x = batch
        # print(f'in valid x shape {x.shape}')
        if self.cfg is not None:
            with torch.no_grad():
                y = batch[0]
                # init condition mask
                cond_mask = torch.tensor(torch.rand(size=(len(x),)) <= self.cfg_drop, device=x.device)
                
                # make conditions
                # print(f'y shape {y.shape}, y max: {torch.max(y)}, y min: {torch.min(y)}')
                c = self.r_model(y)
                
                # print(f'in valid step')
                # print(f'Diff: c max {torch.max(c)} c min {torch.min(c)}')
                
                # set condition mask, if True, set to 0
                c[cond_mask] = torch.ones_like(c[0])
                # repeat this 3 times in channel for CLIP embedding
                c = c.repeat(1,3,1,1)
                
                transform = transforms.Resize(size=(x.shape[2],x.shape[3]),
                                              interpolation=transforms.InterpolationMode.BICUBIC)
                c = transform(c)
                c = torch.clamp(c, 0, 1)
                # print(f'c shape {c.shape}')
                
        
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),),device=self.device)
        
        xt = self.sample_forward(x, t, eps)
        
        eps_theta = self.unet(xt, t, c if self.cfg is not None else None)
        
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
    
    @torch.no_grad()
    def sample_backward_step(self, xt, t, c = None, simple_var=False):
        batch_size = xt.shape[0]
        t_tensor = torch.tensor([t]*batch_size, dtype=torch.long, device=xt.device) # (n, 1)
        eps = self.unet(xt, t_tensor, c)
        
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
    
    @torch.no_grad()
    def sample_backward(self, img, device, simple_var=False, skip_to:Optional[int] = 100):
        self.eval()
        xt: torch.Tensor = img.to(device)
        net = self.unet 
        net = net.to(device)
        print(f'xt shape {xt.shape}')
        if self.cfg is not None:
            # TODO: write cfg guidance diffusion
            # duplicate xt
            xt = torch.concat([xt] * 2, dim=0)
            
            # concat empty condition and c
            c = self.r_model(xt)
            c_concat = torch.concat([torch.ones_like(c), c], dim=0)
            
            # sample
            if skip_to is not None:
                for t in reversed(range(skip_to)):
                    # print(f'ddpm sampling step {t}')
                    xt = self.sample_backward_step(xt, t, c_concat, simple_var)
            else:
                for t in reversed(range(self.n_steps)):
                    # print(f'ddpm sampling step {t}')
                    xt = self.sample_backward_step(xt, t, c_concat, simple_var)
            
            # split xt
            xt_without_cond, xt_with_cond = torch.chunk(xt, 2, dim=0)
            
            # combine them using cfg guidance
            xt = xt_without_cond + self.cfg * (xt_with_cond - xt_without_cond)
            
            return xt
        
        else:
            if skip_to is not None:
                for t in reversed(range(skip_to)):
                    # print(f'ddpm sampling step {t}')
                    xt = self.sample_backward_step(xt, t, simple_var)
            else:
                for t in reversed(range(self.n_steps)):
                    # print(f'ddpm sampling step {t}')
                    xt = self.sample_backward_step(xt, t, simple_var)    
            return xt
    @torch.no_grad()
    def sample_backward_ddim(self, img, net, device, ddim_steps=20, eta=0.0, simple_var=False):
        self.eval()
        with torch.no_grad():
            if simple_var:
                eta = 1
            ts = torch.linspace(self.n_steps - 1, 0, ddim_steps + 1).unsqueeze(1) # size: (ddim_steps + 1)

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
    model = DiffusionModel(unet_config, cfg=3.0)
    print(f'model with cfg? {model.cfg is not None}')
    
    # batch = [torch.randn(32,1,100,100), torch.randn(32, 1, 16, 16)]
    # loss = model.training_step(batch, 0)
    
    # print('loss:',loss)