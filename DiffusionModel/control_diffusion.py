from .diffusion import DiffusionModel
from .control_net import Controlled_UNet
import torch
import pytorch_lightning as pl
import torch.optim as optim
from typing import Optional
import torch.nn.functional as F
from .lion_pytorch import Lion

class ControlDiffusionModel(pl.LightningModule):
    def __init__(self,
                 diffusion_model_dir,
                 reconstruction_model_dir,
                 map_locations='cuda',
                 hint_channels=3):
        super(ControlDiffusionModel, self).__init__()
        self.diffusion_model = DiffusionModel.load_from_checkpoint(diffusion_model_dir, map_location=map_locations)
        self.control_net = Controlled_UNet(reconstruction_model_dir, map_location=map_locations)
        # self.r_model = torch.load(reconstruction_model_dir, map_location=map_locations)
        self.alpha_bars = self.diffusion_model.alpha_bars
    
    # this is a fake forward function which is useless!!!
    def forward(self, x, hint, t):
        return self.control_net(x, t, hint)
    
    @torch.no_grad()
    def sample_forward(self, x0, t, eps=None):
        self.eval()
        with torch.no_grad():
            alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
            if eps is None:
                eps = torch.randn_like(xt)
            else:
                eps = eps
            xt = torch.sqrt(1 - alpha_bar) * eps + torch.sqrt(alpha_bar) * x0
            return xt
    
    @torch.no_grad()
    def sample_backward_step(self, xt, t, hint, simple_var=False, cfg_scale=7.5):
        self.eval()
        batch_size = xt.shape[0]
        t_tensor = torch.tensor([t]*batch_size, dtype=torch.long, device=xt.device) # (n, 1)

         # 条件/无条件预测 (在 step 内部进行)
        eps_uncond = self.diffusion_model.unet(xt, t_tensor)  # 无条件分支
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

        # 时间步循环
        for t in reversed(range(skip_to)):
            print(f'Restoration step {t}', end='\r')
            xt = self.sample_backward_step(xt, t, hint=hint, cfg_scale=cfg_scale) # 调用修改后的 step 函数
        
        return xt
    
    def training_step(self, batch, batch_idx):
        x, hint = batch
        eps = torch.randn_like(x)
        t = torch.randint(0, self.n_steps, (x.size(0),),device=self.device)
        
        xt = self.sample_forward(x, t, eps)
        # print(f'xt shape: {xt.shape}')
        
        eps_theta = self.control_net(xt, t, hint)
        
        # # if epochs >= 10 add final prediction loss else add 0
        # if self.current_epoch >= 100:
        #     final_prediction = self.sample_backward(c,device=self.device, cfg_act=(self.cfg is not None), skip_to=10)
        #     final_loss = F.mse_loss(final_prediction, x)
        #     self.log('final_loss', final_loss, sync_dist=True)
        # else:
        #     final_loss = torch.zeros(1).to(self.device)
        #     self.log('final_loss', final_loss, sync_dist=True)
        
        
        # train_loss = F.mse_loss(eps_theta, eps) + 0.1 * final_loss
        train_loss = F.mse_loss(eps_theta, eps)
        self.log('train_loss', train_loss, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate', lr, on_step=True, sync_dist=True)
        
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x0, hint = batch
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.n_steps, (x0.size(0),),device=self.device)
        
        xt = self.sample_forward(x0, t, eps)
        
        eps_theta = self.control_net(xt, t, hint)
        
        val_loss = F.mse_loss(eps_theta, eps)
        self.log('val_loss', val_loss, sync_dist=True)
        
        return {'val_loss': val_loss, 'x0': x0, 'hint': hint}
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        hint = outputs['hint'].to(self.device)
        x0 = outputs['x0'].to(self.device)
        # Sample from noise
        x0_pred = self.sample_backward(hint, device=self.device, cfg_scale=7.5, skip_to=1000, xt=None)
        
        def normalize_img_0_1(image):
            return (image - image.min()) / (image.max() - image.min())
        
        x_degraded_log = normalize_img_0_1(hint.cpu())
        x_restored_log = normalize_img_0_1(x0_pred.cpu())
        x_gt_log = normalize_img_0_1(x0.cpu())
        
        # log images
        log_images = []
        for i in range(5): # Iterate through batch dimension
            log_images.extend([
                {"image": x_degraded_log[i], "caption": f"Degraded_BatchIdx_{batch_idx}_Image_{i}"},
                {"image": x_restored_log[i], "caption": f"Restored_BatchIdx_{batch_idx}_Image_{i}"},
                {"image": x_gt_log[i], "caption": f"GT_BatchIdx_{batch_idx}_Image_{i}"} # Optional: Log ground truth
            ])

        self.logger.log_images(
            key=f"Validation_Batch_Restoration_BatchIdx_{batch_idx}", # Unique key for each batch
            images=log_images,
            step=self.global_step, # Log at the current global step
        )
        
    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(),lr=1e-4)
        optimizer = Lion(self.parameters(), lr=2e-5, weight_decay=0.1, betas=(0.9,0.95))
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                  mode='min',
        #                                                  factor=0.1,
        #                                                  patience=5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                   T_0=10,
                                                                   T_mult=2)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                    }
                }