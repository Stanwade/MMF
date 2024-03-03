from DiffusionModel.diffusion import DiffusionModel
from datasets import MMFDataset

import torch
from torch.utils.data import DataLoader
from utils import plot_imgs
from torchvision import transforms

# set unet configs
unet_config = {
    'blocks': 2,
    'img_channels': 1,
    'base_channels': 32,
    'ch_mult': [1,2,4,4],
    'norm_type': 'batchnorm',
    'activation': 'mish',
    'with_attn': [True, True, True, False],
    'down_up_sample': False
}

img_size = 32

print('loading model...')
model = DiffusionModel.load_from_checkpoint('DiffusionModel/ckpts2/epoch=199-val_loss=0.0289.ckpt')

target_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])

print('loading datasets...')
a = torch.randn(5,1,img_size,img_size)

out = model.sample_backward(a, model.unet, 'cuda')
# out_ddim = model.sample_backward_ddim(ys, model.unet, 'cuda')
out = out.to('cpu')

plot_imgs(a, name='00a', figsize=(img_size, img_size))
plot_imgs(out, name='00out', figsize=(img_size, img_size))

