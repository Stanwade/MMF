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
    'base_channels': 16,
    'ch_mult': [1,2,4,4],
    'norm_type': 'batchnorm',
    'activation': 'mish',
    'with_attn': [False, False, True, True],
    'down_up_sample': False
}

print('loading model...')
model = DiffusionModel.load_from_checkpoint('DiffusionModel/ckpts2/epoch=9-val_loss=0.0115.ckpt')

target_pipeline = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST)
    ])

print('loading datasets...')
a = torch.randn(5,1,64,64)

out = model.sample_backward(a, model.unet, 'cuda')
# out_ddim = model.sample_backward_ddim(ys, model.unet, 'cuda')
out = out.to('cpu')

plot_imgs(a, name='00a', figsize=(64, 64))
plot_imgs(out, name='00out', figsize=(64, 64))

