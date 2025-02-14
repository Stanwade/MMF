from .diffusion import DiffusionModel
from .control_net import Controlled_UNet
import pytorch_lightning as pl
from typing import Optional
import torch

class Controlled_DiffusionModel():
    def __init__(self, 
                 diffusion_model_dir:str, 
                 map_location:str, 
                 hint_channels:int):
        super().__init__()
        self.diffusion_model = DiffusionModel.load_from_checkpoint(diffusion_model_dir, 
                                                                   map_location=map_location)
        self.control_net = Controlled_UNet(diffusion_model_dir=diffusion_model_dir, 
                                           map_location=map_location, 
                                           hint_channels=hint_channels)
    
    def forward(self, x, condition):
        return self.diffusion_model.forward(x, condition)
    
    @torch.no_grad()
    def sample_forward(self, xT, t, eps = None):
        self.eval()
        with torch.no_grad():
            alpha_bar = self.diffusion_model.alpha_bars[t].reshape(-1, 1, 1, 1)
            if eps is None:
                eps = torch.randn_like(xT)
            else:
                eps = eps
            xt = (torch.sqrt(alpha_bar) * xT + (1 - torch.sqrt(alpha_bar)) * eps)
            return xt
    
    @torch.no_grad()
    def sample_backward_step(self, xt, t, hint, simple_var=False, cfg_scale=7.5):
        self.eval()
        batch_size = xt.shape[0]
        t_tensor = torch.tensor([t]*batch_size, dtype=torch.long, device=xt.device) # (n, 1)

         # 条件/无条件预测 (在 step 内部进行)
        eps_uncond = self.diffusion_model.unet(xt, t_tensor)      # 无条件分支
        eps_cond = self.control_net(xt, t_tensor, hint=hint)      # 条件分支
            
        # CFG实时引导 (在 step 内部进行)
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)


        if t == 0:
            noise = torch.zeros_like(eps)
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(xt) * torch.sqrt(var)

        # DDPM 更新
        mean = (xt - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        xt = mean + noise # 应用噪声
        
        return xt
    
    @torch.no_grad()
    def sample_backward(self, 
                        hint, 
                        device, 
                        cfg_scale=7.5, 
                        skip_to=1000,
                        xt=None):
        """
        Args:
            hint: 控制信号（如边缘图）[B,C,H,W]
        """
        self.eval()
        
        # 初始噪声计算
        if xt is None:
            xt = torch.randn_like(hint).to(device) # 直接在 device 上创建噪声
        else:
            xt = xt.to(device)
        hint = hint.to(device)
        
        #复制 xt 和 hint 以应用 CFG
        xt = torch.cat([xt] * 2, dim=0)
        hint = torch.cat([torch.zeros_like(hint), hint], dim=0)

        # 时间步循环
        for t in reversed(range(skip_to)):
            print(f'Restoration step {t}', end='\r')
            xt = self.sample_backward_step(xt, t, hint=hint, cfg_scale=cfg_scale) # 调用修改后的 step 函数
        
        return xt.clamp(-1,1)