from ReconstructionModel.model import ReconstructionModel
from datasets import MMFDataset, MMFGrayScaleDataset, MMFMNISTDataset
from DiffusionModel.diffusion import DiffusionModel

import torch
from torch.utils.data import DataLoader
from utils import plot_imgs
from torchvision import transforms


print('loading model...')
model = ReconstructionModel.load_from_checkpoint('ReconstructionModel/ckpts_mnist/epoch=49-val_loss=0.0175.ckpt')

print('loading datasets...')
dataset = MMFMNISTDataset(root='datasets', train=False)

test_loader = DataLoader(dataset=dataset,
                         batch_size=5,
                         shuffle=False)

inputs, labels = next(iter(test_loader))
print(f'inputs size {inputs.shape}')

out = model(inputs.cuda())
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
model = DiffusionModel.load_from_checkpoint('./DiffusionModel/ckpts_mnist/epoch=49-val_loss=0.0101.ckpt')

target_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    ])

print('loading datasets...')

out = target_pipeline(out.float())

out = model.sample_backward(out, model.unet, 'cuda', skip=True, skip_to=10)
# out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu')
out = (out + 1) / 2 * 255
out = out.clamp(0,255).to(torch.uint8)


plot_imgs(out, name='01out2')
# plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

