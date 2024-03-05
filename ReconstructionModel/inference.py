from ReconstructionModel.model import ReconstructionModel
from datasets import MMFDataset, MMFGrayScaleDataset

import torch
from torch.utils.data import DataLoader
from utils import plot_imgs
from torchvision import transforms


img_size = 16

print('loading model...')
model = ReconstructionModel.load_from_checkpoint('ReconstructionModel/ckpts/epoch=719-val_loss=0.0629.ckpt')

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

out = model(inputs.cuda())
# out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu').detach()
out = (out + 1) / 2 * 255
out = out.clamp(0,255).to(torch.uint8)


print(f'out max {torch.max(out)}')
print(f'out min {torch.min(out)}')

plot_imgs(inputs, name='01a', figsize=(img_size, img_size))
plot_imgs(labels, name='01label', figsize=(img_size, img_size))
plot_imgs(out, name='01out', figsize=(img_size, img_size))
# plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

