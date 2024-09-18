from ReconstructionModel.model import ReconstructionModel
from datasets import MMF_imgNet32Dataset
from DiffusionModel.diffusion import DiffusionModel

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import plot_imgs, calculate_ssim
from torchvision import transforms
import numpy as np

print('loading model...')
model = ReconstructionModel.load_from_checkpoint('ReconstructionModel/ckpts_imgnet32_real/epoch=99-val_loss=0.0078-v1.ckpt',map_location='cuda')

print('loading datasets...')
dataset = MMF_imgNet32Dataset(root='datasets', train=False)

test_loader = DataLoader(dataset=dataset,
                         batch_size=5*3,
                         shuffle=False)

inputs, labels = next(iter(test_loader))
print(f'inputs size {inputs.shape}')


out = model(inputs.cuda())
# out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu').detach()
print(f'out size {out.shape}')
# out = (out + 1) / 2 * 255
# out = out.clamp(0,255).to(torch.uint8)

inputs_rgb = []
labels_rgb = []
outs_rgb = []

for i in range(5):
    inputs_rgb.append(np.expand_dims(np.concatenate([inputs[3*i],inputs[3*i+1],inputs[3*i+2]]),axis=0)/255)
    labels_rgb.append(np.expand_dims(np.concatenate([labels[3*i],labels[3*i+1],labels[3*i+2]]),axis=0))
    outs_rgb.append(np.expand_dims(np.concatenate([out[3*i],out[3*i+1],out[3*i+2]]),axis=0))

inputs_rgb = torch.from_numpy(np.concatenate(inputs_rgb)).permute(0,2,3,1)
labels_rgb = torch.from_numpy(np.concatenate(labels_rgb)).permute(0,2,3,1)
outs_rgb = torch.from_numpy(np.concatenate(outs_rgb)).permute(0,2,3,1)

print(f'inputs rgb shape {inputs_rgb.shape}')
print(f'labels rgb shape {labels_rgb.shape}')
print(f'outs rgb shape {labels_rgb.shape}')

plot_imgs(inputs_rgb, name='01inputs',cmap=None)
plot_imgs(labels_rgb, name='01label',cmap=None)
plot_imgs(outs_rgb, name='01out1',cmap=None)
# plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

# * from here is Diffusion part
img_size = 32

print('loading model...')
model = DiffusionModel.load_from_checkpoint('./DiffusionModel/ckpts_imgnet32/epoch=69-val_loss=0.0228.ckpt', map_location="cuda")

target_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])

print('loading datasets...')

out = target_pipeline(out.float())

out = model.sample_backward(out, 'cuda', skip=True, skip_to=8)
# out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu')
out = out  * 255
out = out.clamp(0,255).to(torch.uint8)

ssims_list = []

label_resize = transforms.Resize((out.shape[-1]),interpolation=transforms.InterpolationMode.NEAREST)
labels_rgb_resize = (label_resize(labels_rgb.permute(0,3,1,2)) * 255).clamp(0,255).to(torch.uint8)

print(f'out max = {torch.max(out)}')
print(f'out min = {torch.min(out)}')
print(f'label max = {torch.max(labels_rgb_resize)}')
print(f'label min = {torch.min(labels_rgb_resize)}')
print(f'label shape {labels_rgb_resize.shape}')

outs2_rgb = []

for i in range(5):
    outs2_rgb.append(np.expand_dims(np.concatenate([out[3*i],out[3*i+1],out[3*i+2]]),axis=0))
outs2_rgb = torch.from_numpy(np.concatenate(outs2_rgb))
print(f'outs2 rg shape {outs2_rgb.shape}')

for i in range(outs2_rgb.shape[0]):
    ssims_list.append(calculate_ssim(outs2_rgb[i].float(), labels_rgb_resize[i].float()))

outs2_rgb = outs2_rgb.permute(0,2,3,1)

plot_imgs(outs2_rgb, name='01out2',str_list=ssims_list)
# plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

noise = torch.randn((5,1,32,32))
plot_imgs(noise, name='01noise')

out = model.sample_backward(noise, 'cuda', skip=False).to('cpu')

plot_imgs(out, name='01out3')

