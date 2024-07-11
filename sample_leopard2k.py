from ReconstructionModel.model import ReconstructionModel
from datasets import MMF_imgNet32Dataset
from DiffusionModel.diffusion import DiffusionModel
from UNetModel.model import UNetModel

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import plot_imgs, calculate_ssim, restore_images_from_patches
from torchvision import transforms
import numpy as np
from tqdm import trange

print('loading model...')
model = ReconstructionModel.load_from_checkpoint('ReconstructionModel/ckpts_leopard2k/epoch=79-val_loss=0.0630.ckpt')
# model = UNetModel.load_from_checkpoint('UNetModel/ckpts_leopard2k/epoch=19-val_loss=0.0041.ckpt').cuda(0)
print(torch.cuda.memory_allocated(0) / 1024)


print('loading datasets...')
inputs = torch.from_numpy(np.load('/home/wty/mmf_-demo/datasets/leopard_2k/speckles_merged.npy')).unsqueeze(1).float()

labels = torch.from_numpy(np.load('/home/wty/mmf_-demo/datasets/leopard_2k/pattern_gray.npy')).reshape(-1,32,32).unsqueeze(1)

print(f'inputs size {inputs.shape}')
print(f'labels size {labels.shape}')

inputs_list = []
outputs_list = []
batch_size = 64
for i in trange(inputs.shape[0] // batch_size + 1):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, inputs.size(0))
    batch_inputs = inputs[start_idx:end_idx].cuda(0)
    # print(f'loop {i}: memory: {torch.cuda.memory_allocated(0) / (1024*1024)} MB')
    with torch.no_grad():
        outputs_list.append(model(batch_inputs).to('cpu'))
out = torch.from_numpy(np.concatenate(outputs_list))

print(f'len of output list {len(outputs_list)}, outputs shape {out.shape}')

resize = transforms.Resize((32,32),interpolation=transforms.InterpolationMode.NEAREST)
# out = model(inputs.cuda(0))
out = torch.clamp(out,0,1)
out = resize(out)
print(f'out shape {out.shape}; out max {torch.max(out):.2f}, out min {torch.min(out):.2f}')
# # out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
out = out.to('cpu').detach()
# # out = (out + 1) / 2 * 255
# # out = out.clamp(0,255).to(torch.uint8)

plot_imgs(inputs, name='00inputs',cmap='gray')
plot_imgs(labels, name='00label',cmap='gray')
plot_imgs(out, name='00out1',cmap='gray')

inputs_rgb = []
labels_rgb = []
outs_rgb = []

for i in range(int(out.shape[0]/3)):
    inputs_rgb.append(np.expand_dims(np.concatenate([inputs[3*i],inputs[3*i+1],inputs[3*i+2]]),axis=0)/255)
    labels_rgb.append(np.expand_dims(np.concatenate([labels[3*i],labels[3*i+1],labels[3*i+2]]),axis=0)/255)
    outs_rgb.append(np.expand_dims(np.concatenate([out[3*i],out[3*i+1],out[3*i+2]]),axis=0))

inputs_rgb = torch.from_numpy(np.concatenate(inputs_rgb)).permute(0,2,3,1)
labels_rgb = torch.from_numpy(np.concatenate(labels_rgb)).permute(0,2,3,1)
outs_rgb = torch.from_numpy(np.concatenate(outs_rgb)).permute(0,2,3,1)

print(f'inputs rgb shape {inputs_rgb.shape}')
print(f'labels rgb shape {labels_rgb.shape}, labels max {torch.max(labels_rgb)}, labels min {torch.min(labels_rgb)}')
print(f'outs rgb shape {labels_rgb.shape}')

plot_imgs(inputs_rgb, name='01inputs',cmap=None)
plot_imgs(labels_rgb, name='01label',cmap=None)
plot_imgs(outs_rgb, name='01out1',cmap=None)

big_img = restore_images_from_patches(outs_rgb.permute(0,3,1,2),1600,2560).squeeze()
big_img = (big_img.transpose(1,2,0) * 255).astype(np.uint8)
print(f'big img shape {big_img.shape}, big img max {np.max(big_img)}')

import cv2
big_img = cv2.cvtColor(big_img,cv2.COLOR_BGR2RGB)
cv2.imwrite('imgs/big_img.jpg',big_img)

big_img_label = restore_images_from_patches(labels_rgb.permute(0,3,1,2),1600,2560).squeeze()
big_img_label = (big_img_label.transpose(1,2,0) * 255).astype(np.uint8)
big_img_label = cv2.cvtColor(big_img_label,cv2.COLOR_BGR2RGB)
cv2.imwrite('imgs/big_img_label.jpg',big_img_label)


# # plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

# # * from here is Diffusion part
# img_size = 32

# print('loading model...')
# model = DiffusionModel.load_from_checkpoint('./DiffusionModel/ckpts_imgnet32/epoch=59-val_loss=0.0266.ckpt')

# target_pipeline = transforms.Compose([
#         transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
#     ])

# print('loading datasets...')

# out = target_pipeline(out.float())

# out = model.sample_backward(out, model.unet, 'cuda:4', skip=True, skip_to=8)
# # out_ddim = model.sample_backward_ddim(a, model.unet, 'cuda')
# out = out.to('cpu')
# out = out  * 255
# out = out.clamp(0,255).to(torch.uint8)

# ssims_list = []

# label_resize = transforms.Resize((out.shape[-1]),interpolation=transforms.InterpolationMode.NEAREST)
# labels_rgb_resize = (label_resize(labels_rgb.permute(0,3,1,2)) * 255).clamp(0,255).to(torch.uint8)

# print(f'out max = {torch.max(out)}')
# print(f'out min = {torch.min(out)}')
# print(f'label max = {torch.max(labels_rgb_resize)}')
# print(f'label min = {torch.min(labels_rgb_resize)}')
# print(f'label shape {labels_rgb_resize.shape}')

# outs2_rgb = []

# for i in range(5):
#     outs2_rgb.append(np.expand_dims(np.concatenate([out[3*i],out[3*i+1],out[3*i+2]]),axis=0))
# outs2_rgb = torch.from_numpy(np.concatenate(outs2_rgb))
# print(f'outs2 rg shape {outs2_rgb.shape}')

# for i in range(outs2_rgb.shape[0]):
#     ssims_list.append(calculate_ssim(outs2_rgb[i].float(), labels_rgb_resize[i].float()))

# outs2_rgb = outs2_rgb.permute(0,2,3,1)

# plot_imgs(outs2_rgb, name='01out2',str_list=ssims_list)
# # plot_imgs(out_ddim, name='00outddim', figsize=(img_size, img_size))

# noise = torch.randn((5,1,32,32))
# plot_imgs(noise, name='01noise')

# out = model.sample_backward(noise, model.unet, 'cuda:4', skip=False).to('cpu')

# plot_imgs(out, name='01out3')

