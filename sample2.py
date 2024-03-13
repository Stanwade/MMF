from ReconstructionModel.model import ReconstructionModel
from datasets import MMFDataset, MMFGrayScaleDataset
from DiffusionModel.diffusion import DiffusionModel

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import plot_imgs, calculate_ssim
from torchvision import transforms


img_size = 16

print('loading model...')
model = ReconstructionModel.load_from_checkpoint('ReconstructionModel/ckpts/epoch=149-val_loss=0.0630.ckpt')

target_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])

print('loading datasets...')
dataset = MMFGrayScaleDataset(root='datasets', train=False)

test_loader = DataLoader(dataset=dataset,
                         batch_size=5,
                         shuffle=False)

inputs, labels = next(iter(test_loader))
print(f'inputs size {inputs.shape}')

out = model(inputs.cuda() * 3)
# out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu').detach()
# out = (out + 1) / 2 * 255
# out = out.clamp(0,255).to(torch.uint8)

plot_imgs(inputs, name='01inputs')
plot_imgs(labels, name='01label')
plot_imgs(out, name='01out1')
# plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

img_size = 48

print('loading model...')
model = DiffusionModel.load_from_checkpoint('./DiffusionModel/ckpts_grayscale/epoch=59-val_loss=0.0217.ckpt')

target_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])

print('loading datasets...')

out = target_pipeline(out.float() / 3)

out = model.sample_backward(out, model.unet, 'cuda', skip=True, skip_to=10)
# out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu')
out = out * 255
out = out.clamp(0,255).to(torch.uint8)

ssims_list = []

labels_resize = F.interpolate(labels.unsqueeze(0), out.shape[1:], mode='nearest').squeeze(0) * 255

print(f'out max = {torch.max(out)}')
print(f'out min = {torch.min(out)}')
print(f'label max = {torch.max(labels_resize)}')
print(f'label min = {torch.min(labels_resize)}')

for i in range(out.shape[0]):
    ssims_list.append(calculate_ssim(out[i].float(), labels_resize[i].float()))


plot_imgs(out, name='01out2', str_list=ssims_list)
# plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

