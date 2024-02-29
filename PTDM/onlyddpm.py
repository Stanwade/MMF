from torch.utils.data import DataLoader
from datasets import MMFDataset
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from PTDM.networks import UNet

class ddpm():
    def __init__(self,
                 n_steps: int = 1000,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        
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
    
    
    last_memory = 0
    
    def get_memory_total():
        global last_memory
        last_memory = torch.cuda.memory_allocated() / 1024 / 1024
        return last_memory
    
    def get_memory_diff():
        last = last_memory
        total = get_memory_total()
        return total - last, total
    
    unet_config = {
        'blocks': 2,
        'img_channels': 1,
        'base_channels': 64,
        'ch_mult': [1,2,4,4],
        'norm_type': 'batchnorm',
        'activation': 'mish',
        'with_attn': [False,False,False,True],
        'down_up_sample': False
    }
    unet = UNet(**unet_config).cuda()
    myddpm = ddpm(1000,0.0001,0.02)
    
    
    print(f'model to cuda: {get_memory_diff()}')
    epochs = 100
    
    # Load data
    print('lodaing data')
    train_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=True)
    validation_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=False)
    # create loader
    print('creating data loader')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)     
    
    # optimizer
    print('creating optimizer')
    optimizer = optim.AdamW(unet.parameters(), lr=1e-4)
    

    
    # train loop
    for i in range(epochs):
        print(f'running epoch {i}')
        tick = time.time()
        total_loss = 0
        for batch in train_loader:
            x = batch[1].to('cuda')
            print(f'x size: {x.shape}')
            print(f'epoch {i} batch loaded: {get_memory_diff()}')
            
            batchsize = x.size(0)
            eps = torch.randn_like(x).to('cuda')
            print(f'epoch {i} eps loaded: {get_memory_diff()}')
            t = torch.randint(0, myddpm.n_steps, (x.size(0),)).to('cuda')
            xt = myddpm.sample_forward_step(x, t, eps)
            print(f'xt shape: {xt.shape}; t shape: {t.shape}')
            
            print('tring to predict...')
            eps_theta = unet(xt, t)
            print(f'epoch {i} unet inference and intermediate: {get_memory_diff()}')
            train_loss = F.mse_loss(eps_theta, eps)
            exit('debug finished')
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += train_loss.item() * batchsize
        total_loss /= len(train_loader.dataset)
        toc = time.time()
        print(f'epoch {i}, loss: {total_loss}, time: {toc-tick}')
            # self.log('train_loss', train_loss, sync_dist=True)