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
import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from .diffusion import DiffusionModel
from .control_net import Controlled_UNet
from .lion_pytorch import Lion

class ControlDiffusionModel(pl.LightningModule):
    def __init__(self,
                 diffusion_model_dir,
                 reconstruction_model_dir,
                 map_locations='cuda',
                 hint_channels=3):
        super(ControlDiffusionModel, self).__init__()
        
        # 加载基础DiffusionModel(内部包含 betas, alphas, alpha_bars 等)
        self.diffusion_model = DiffusionModel.load_from_checkpoint(diffusion_model_dir, 
                                                                   map_location=map_locations)
        # ControlNet，用于条件分支
        self.control_net = Controlled_UNet(reconstruction_model_dir, 
                                           map_location=map_locations,
                                           hint_channels=hint_channels)
        
        # 从diffusion_model中获取关键参数
        self.alpha_bars = self.diffusion_model.alpha_bars
        self.betas = self.diffusion_model.betas
        self.alphas = self.diffusion_model.alphas
        self.n_steps = self.diffusion_model.n_steps
        
    # 仅用于接口占位，无实际计算意义
    def forward(self, x, hint, t):
        return self.control_net(x, t, hint)
    
    @torch.no_grad()
    def sample_forward(self, x0, t, eps=None):
        """
        将干净图像x0从time=0扩散到time=t得到xt。
        """
        self.eval()
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x0)  # 修复：原来没有定义xt就调用torch.randn_like(xt)
        # 计算x_t
        xt = torch.sqrt(1 - alpha_bar) * eps + torch.sqrt(alpha_bar) * x0
        return xt
    
    @torch.no_grad()
    def sample_backward_step(self, xt, t, hint, simple_var=False, cfg_scale=7.5):
        """
        DDPM单步逆过程采样(带CFG引导)。
        """
        self.eval()
        batch_size = xt.shape[0]
        t_tensor = torch.tensor([t]*batch_size, dtype=torch.long, device=xt.device)

        # 条件/无条件预测噪声
        eps_uncond = self.diffusion_model.unet(xt, t_tensor)     # 无条件分支
        eps_cond   = self.control_net(xt, t_tensor, hint=hint)   # 条件分支
        
        # CFG 计算
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # 计算 variance
        if t == 0:
            noise = torch.zeros_like(eps)
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(xt) * torch.sqrt(var)

        # DDPM公式: x_{t-1} = (x_t - (1 - alpha_t)/sqrt(1-alpha_bar_t)*eps)/sqrt(alpha_t) + sqrt(var)*noise
        mean = (xt - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        xt = mean + noise
        return xt
    
    @torch.no_grad()
    def sample_backward(self, 
                        hint, 
                        device, 
                        cfg_scale=7.5, 
                        skip_to=1000,
                        xt=None):
        """
        逐步DDPM逆过程采样。
        Args:
            hint: 控制信号（如边缘图）
            skip_to: 默认假设扩散到 1000 步（或 self.n_steps）。可以减少。
            xt: 如果传入初始噪声，则使用之；否则随机采样。
        """
        self.eval()
        if xt is None:
            xt = torch.randn_like(hint).to(device)
        else:
            xt = xt.to(device)
        hint = hint.to(device)

        # 时间步循环
        for t in reversed(range(skip_to)):
            # 逐步逆扩散
            xt = self.sample_backward_step(xt, t, hint=hint, cfg_scale=cfg_scale)
        return xt

    # ---------------------------------------
    #   DDIM 加速采样相关函数
    # ---------------------------------------
    @torch.no_grad()
    def sample_backward_step_ddim(self, xt, t, t_next, hint, cfg_scale=7.5, eta=0.0):
        """
        DDIM 单步采样(支持CFG)。 
        其中 t > t_next，一般是从大到小的顺序。
        Args:
            xt: 当前时刻的张量 x_t
            t: 当前时刻
            t_next: 下一时刻(通常 t_next = t-跳步数)
            hint: 控制信号
            cfg_scale: CFG强度
            eta: DDIM中的噪声系数, 0即无随机性(确定性DDIM)
        """
        self.eval()
        batch_size = xt.shape[0]
        t_tensor = torch.tensor([t]*batch_size, dtype=torch.long, device=xt.device)

        # 1. 预测噪声 (包含CFG)
        eps_uncond = self.diffusion_model.unet(xt, t_tensor)
        eps_cond   = self.control_net(xt, t_tensor, hint=hint)
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # 2. 提取alpha_bar
        alpha_bar_t     = self.alpha_bars[t]
        alpha_bar_t_next = self.alpha_bars[t_next] if t_next >= 0 else self.alpha_bars[0]

        # 3. 计算 x0 (去噪后图像)
        #    x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        x0 = (xt - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        # DDIM 公式（eta=0 即无随机性）
        # x_{t_next} = sqrt(alpha_bar_{t_next}) * x0 + sqrt(1 - alpha_bar_{t_next}) * eps + 额外噪声项
        # 额外噪声项: sigma = eta * sqrt((1 - alpha_bar_t_next)/(1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t/alpha_bar_t_next)
        # （不需要非常严格，可灵活调整）
        
        # 4. 计算 sigma (若想要确定性DDIM，可以直接设 eta = 0)
        sigma = 0
        if eta > 0.0 and t_next >= 0:
            # 额外添加一些随机性
            alpha_ratio = (1 - alpha_bar_t_next) / (1 - alpha_bar_t)
            sigma = eta * torch.sqrt(alpha_ratio * (1 - alpha_bar_t / alpha_bar_t_next))

        # 5. 组装 x_{t_next}
        # 这时如果 sigma>0，还要加上 sigma * 随机噪声
        noise = torch.randn_like(xt) if (sigma > 0) else 0.0
        xt_next = torch.sqrt(alpha_bar_t_next) * x0 + torch.sqrt(1 - alpha_bar_t_next) * eps + sigma * noise

        return xt_next

    @torch.no_grad()
    def sample_backward_ddim(self, 
                             hint, 
                             device, 
                             cfg_scale=7.5, 
                             skip_to=1000, 
                             steps=50, 
                             eta=0.0,
                             xt=None):
        """
        使用 DDIM 进行加速采样，通过跳步减少迭代次数。
        Args:
            hint:       控制信号
            skip_to:    从几步(比如1000)开始逆推
            steps:      需要多少步完成从 skip_to -> 0 的采样
            eta:        DDIM中的噪声因子, 0表确定性采样
        """
        self.eval()
        if xt is None:
            xt = torch.randn_like(hint).to(device)
        else:
            xt = xt.to(device)
        hint = hint.to(device)

        # 构造一个从 skip_to-1 到 0 的等间隔序列 (含首尾)
        # 比如 skip_to=1000, steps=50, 则在[999, 0]之间均匀取51个点
        # times[0] = 999, times[-1] = 0, 中间每次递减 999/(50) ~ 19.98
        times = torch.linspace(skip_to - 1, 0, steps + 1, dtype=torch.long, device=xt.device)

        for i in range(steps):
            t = times[i].item()
            t_next = times[i+1].item()
            xt = self.sample_backward_step_ddim(xt, 
                                                t=int(t), 
                                                t_next=int(t_next), 
                                                hint=hint, 
                                                cfg_scale=cfg_scale,
                                                eta=eta)
        return xt
    # ---------------------------------------
    
    def training_step(self, batch, batch_idx):
        x, hint = batch
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),), device=self.device)
        
        xt = self.sample_forward(x, t, eps)
        
        # 预测出的噪声
        eps_theta = self.control_net(xt, t, hint)
        train_loss = F.mse_loss(eps_theta, eps)
        
        self.log('train_loss', train_loss, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, sync_dist=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        x0, hint = batch
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.n_steps, (x0.size(0),), device=self.device)
        
        xt = self.sample_forward(x0, t, eps)
        eps_theta = self.control_net(xt, t, hint)
        
        val_loss = F.mse_loss(eps_theta, eps)
        self.log('val_loss', val_loss, sync_dist=True)
        
        return {'val_loss': val_loss, 'x0': x0, 'hint': hint}
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        """
        每个验证batch结束后，进行一次采样并可视化结果。
        """
        hint = outputs['hint'].to(self.device)
        x0 = outputs['x0'].to(self.device)
        
        # 1. DDPM 完整采样
        x0_pred_ddpm = self.sample_backward(hint, 
                                            device=self.device, 
                                            cfg_scale=7.5, 
                                            skip_to=self.n_steps, 
                                            xt=None)

        # 2. DDIM 加速采样(例如50步)
        x0_pred_ddim = self.sample_backward_ddim(hint,
                                                 device=self.device,
                                                 cfg_scale=7.5,
                                                 skip_to=self.n_steps,
                                                 steps=50,      # 仅示例，可自行调整
                                                 eta=0.0,       # 0为确定性
                                                 xt=None)
        
        # 归一化到[0,1]用于可视化
        def normalize_img_0_1(image):
            return (image - image.min()) / (image.max() - image.min())
        
        x_hint_log = normalize_img_0_1(hint.cpu())
        x_ddpm_log = normalize_img_0_1(x0_pred_ddpm.cpu())
        x_ddim_log = normalize_img_0_1(x0_pred_ddim.cpu())
        x_gt_log   = normalize_img_0_1(x0.cpu())
        
        # 仅可视化前5个
        log_images = []
        for i in range(min(5, hint.size(0))):
            log_images.extend([
                {"image": x_hint_log[i], "caption": f"Hint_BatchIdx_{batch_idx}_Img_{i}"},
                {"image": x_ddpm_log[i], "caption": f"DDPM_BatchIdx_{batch_idx}_Img_{i}"},
                {"image": x_ddim_log[i], "caption": f"DDIM_BatchIdx_{batch_idx}_Img_{i}"},
                {"image": x_gt_log[i],   "caption": f"GT_BatchIdx_{batch_idx}_Img_{i}"}
            ])

        # 将图像日志记录到logger
        self.logger.log_images(
            key=f"Validation_Batch_{batch_idx}",
            images=log_images,
            step=self.global_step,
        )
        
    def configure_optimizers(self):
        # 例子：使用Lion优化器
        optimizer = Lion(self.parameters(), lr=2e-5, weight_decay=0.1, betas=(0.9,0.95))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                   T_0=10,
                                                                   T_mult=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
