from diffusion import DiffusionModel
from datasets import MMFDataset

import torch
from torch.utils.data import DataLoader
from utils import plot_imgs

# set unet configs
unet_config = {
    'blocks': 2,
    'img_channels': 1,
    'base_channels': 16,
    'ch_mult': [1,2,4,4],
    'norm_type': 'batchnorm',
    'activation': 'mish',
    'with_attn': [False, False, True, True],
    'down_up_sample': False
}

model = DiffusionModel(unet_config)
model = model.load_from_checkpoint('ckpt/diffusion_model.ckpt')

dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=False)

loader = DataLoader(dataset, batch_size=32, shuffle=False)
xs, ys = next(iter(loader))

out = model.sample_backward(xs, model.unet, 'cuda')
out_ddim = model.sample_backward_ddim(xs, model.unet, 'cuda')

plot_imgs(xs, name='input')
plot_imgs(ys, name='target')
plot_imgs(out, name='output')