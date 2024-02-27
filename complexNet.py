import torch
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def channels_to_complex(real, imag):
    return torch.complex(real, imag)

def complex_to_channels(z):
    return torch.real(z), torch.imag(z)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.real_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))
    
    def forward(self, input):
        real = F.linear(torch.real(input), self.real_weight) - F.linear(torch.imag(input), self.imag_weight)
        imag = F.linear(torch.imag(input), self.real_weight) + F.linear(torch.real(input), self.imag_weight)
        return torch.complex(real, imag)

if __name__ == '__main__':
    # 假设的复数输入数据
    real_input = torch.randn(5, 3)
    imag_input = torch.randn(5, 3)
    complex_input = torch.complex(real_input, imag_input)

    # 创建ComplexLinear层实例
    complex_linear = ComplexLinear(in_features=3, out_features=2)

    # 前向传播
    output = complex_linear(complex_input)
    print(output)
