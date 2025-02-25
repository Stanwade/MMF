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
    'base_channels': 10,
    'ch_mult': [1,2,4,4],
    'norm_type': 'batchnorm',
    'activation': 'mish',
    'with_attn': [True, True, True, False],
    'down_up_sample': False
}

img_size = 48

print('loading model...')
model = DiffusionModel.load_from_checkpoint('./DiffusionModel/ckpts_grayscale/epoch=39-val_loss=0.0231.ckpt')

target_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])

print('loading datasets...')
a = torch.randn(5,1,img_size,img_size)

out = model.sample_backward(a, model.unet, 'cuda', n_steps=1000, skip=False)
# out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu')
out = (out + 1) / 2 * 255
out = out.clamp(0,255).to(torch.uint8)


print(f'out max {torch.max(out)}')
print(f'out min {torch.min(out)}')

plot_imgs(a, name='00a', figsize=(img_size, img_size))
plot_imgs(out, name='00out', figsize=(img_size, img_size))
# plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

