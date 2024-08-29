import torch
import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self,label_size:int=32, input_channel:int = 1):
        super(VGGNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, label_size*label_size)  # Output is a 32x32 vector
        )
        self.lable_size = label_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.view(-1, 1, self.lable_size, self.lable_size)

def createVGGBlock(in_channels, out_channels, num_blocks):
    layers = []
    for _ in range(num_blocks):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(VGGBlock, self).__init__()
        self.layers = createVGGBlock(in_channels, out_channels, num_blocks)

    def forward(self, x):
        x = self.layers(x)
        return x

class VGGConvBlocks(nn.Module):
    def __init__(self,
                 input_channel:int,
                 channel_list:list,
                 blocks:list) -> None:
        super(VGGConvBlocks,self).__init__()
        self.conv_layers = nn.ModuleList([])
        for i in range(len(blocks)):
            self.conv_layers.append(VGGBlock(input_channel,channel_list[i],blocks[i]))
            input_channel = channel_list[i]
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

class CustomVGG(nn.Module):
    def __init__(self,label_size:int = 32,
                 input_channel:int = 1, 
                 input_size:int = 32,
                 channel_list_conv:list = [64, 128, 256], 
                 blocks:list = [2, 2, 3]):
        if len(channel_list_conv) != len(blocks):
            raise ValueError(f"channel_list and blocks must be the same length, got {len(channel_list_conv)} and {len(blocks)}")
        super(CustomVGG, self).__init__()
        
        self.conv_layers = VGGConvBlocks(input_channel=input_channel,
                                         channel_list=channel_list_conv,
                                         blocks=blocks)
        
        size = input_size // (2**len(blocks))
        print(f'size {size}, fc input channel{channel_list_conv[-1]}')
        self.fc_layers = nn.Sequential(
            nn.Linear(channel_list_conv[-1] * size * size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, label_size*label_size)  # Output is a 32x32 vector
        )
        self.lable_size = label_size

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layers(x)
        return x.view(-1, 1, self.lable_size, self.lable_size)