import torch
from torch import nn
from datasets import MMFDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from networks import FullyConnectedNetwork, OneLayer
import matplotlib.pyplot as plt
from utils import plot_imgs

from DiffusionModel.diffusion import DiffusionModel

OneLayer = False

# load model from pth file
if OneLayer:
    model = OneLayer()
else:
    model = FullyConnectedNetwork()

# 加载模型状态字典
state_dict = torch.load('model_checkpoint_500.pth', 
                        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# 调整状态字典中的键，去除'module.'前缀
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 加载调整后的状态字典到模型
model.load_state_dict(new_state_dict)

# load validation dataset
valid_dataset = MMFDataset(root='./datasets', train=False)

# create data loader
valid_loader = DataLoader(valid_dataset, batch_size=5, shuffle=False)

# get inputs from the validation dataset
inputs, labels = next(iter(valid_loader))

# plot inputs
plot_imgs(inputs, name='inputs')

# plot labels
plot_imgs(labels, name='labels')

# run inference
model.eval()
with torch.no_grad():
    outputs1 = model(inputs)
    
# plot outputs
plot_imgs(outputs1, name='outputs_1')


# Second Model

print('loading diffusion model...')
model = DiffusionModel.load_from_checkpoint('DiffusionModel/ckpts3/epoch=59-val_loss=0.0293.ckpt')

resize_16_64 = transforms.Resize((32,32),interpolation=transforms.InterpolationMode.NEAREST_EXACT)
xs_2nd = resize_16_64(outputs1)



outputs2 = model.sample_backward(xs_2nd, model.unet, 'cuda', skip_to=10)
# out_ddim = model.sample_backward_ddim(ys, model.unet, 'cuda')
outputs2 = outputs2.to('cpu')

outputs2 = (outputs2 + 1) / 2 * 255
out = outputs2.clamp(0,255).to(torch.uint8)

plot_imgs(outputs2,name='outputs_2')

# i = 0

# for xt in outputs2:
#     xt = xt.to('cpu')
#     plot_imgs(xt, name=f'outputs_2_{i}', figsize=(32, 32))
#     i += 1