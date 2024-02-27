import torch
from torch import nn
from datasets import MMFDataset
from torch.utils.data import DataLoader
from networks import FullyConnectedNetwork, OneLayer
import matplotlib.pyplot as plt
from utils import plot_imgs

OneLayer = False

# load model from pth file
if OneLayer:
    model = OneLayer()
else:
    model = FullyConnectedNetwork()

# 加载模型状态字典
state_dict = torch.load('model_checkpoint_100.pth', 
                        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# 调整状态字典中的键，去除'module.'前缀
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 加载调整后的状态字典到模型
model.load_state_dict(new_state_dict)

# load validation dataset
valid_dataset = MMFDataset(root='./datasets/100m_200/16x16/1', train=False)

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
    outputs = model(inputs)
    
# plot outputs
plot_imgs(outputs, name='outputs')