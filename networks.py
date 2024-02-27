import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import torch.backends.cudnn as cudnn


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x):
        time = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
        emb = torch.sin(time / (10000 ** (torch.arange(0, 2 * self.embedding_dim, 2) / self.embedding_dim)))
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connect = nn.Sequential()
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip_connect(x)
        out = F.relu(out)
        return out

class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.norm1 = nn.BatchNorm2d(1)
        self.flatten = nn.Flatten()
        self.linear_act_stack = nn.Sequential(
            nn.Linear(1*100*100, 512),
            nn.Mish(),
            nn.Linear(512, 1*16*16),
            nn.Mish()
        )

    def forward(self, x):
        x = self.norm1(x)
        x = self.flatten(x)
        logits = self.linear_act_stack(x)
        return logits.view(-1, 1, 16, 16)


class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.fc1 = nn.Linear(100*100, 16*16)

    def forward(self, x):
        x = x.view(-1, 100*100)
        x = self.fc1(x)
        return x.view(-1, 16, 16)


    
if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    model = ResBlock(3, 3)
    out = model(x)
    b,c = torch.chunk(out, 2, dim=2)
    print(b.size(), c.size())
    teb = TimeEmbedding(3)
    print(x)
    out = teb(x)
    print(out.size())
    print(out)