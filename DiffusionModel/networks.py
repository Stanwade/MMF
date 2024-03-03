# Residual block
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Optional, Union, List
from DiffusionModel.utils import default, create_norm, create_activation



class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        half_dim = embedding_dim // 8
        self.emb = torch.exp(torch.arange(0, half_dim) * -math.log(max_period) / half_dim)
        self.emb_const: torch.Tensor
        self.register_buffer("emb_const", self.emb[None, :])
        
        self.proj = nn.Sequential(
            nn.Linear(self.embedding_dim // 4, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.emb_const * t[:, None]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = self.proj(emb)
        return emb
        

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        assert d_model % 2 == 0
        
        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)
        
    def forward(self, t):
        return self.embedding(t)


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_channels,
                 stride=1,
                 bias=False,
                 activation="lrelu",
                 norm_type="batchnorm"):
        super(ResBlock, self).__init__()
        self.norm1 = create_norm(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm2 = create_norm(out_channels, norm_type)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        if in_channels != out_channels or stride != 1:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connect = nn.Identity()

        self.time_emb = nn.Linear(time_emb_channels, out_channels*2)
        nn.init.constant_(self.time_emb.weight, 0)
        nn.init.constant_(self.time_emb.bias, 0)
        
        self.act = create_activation(activation)
        
    def forward(self, x, t_emb: torch.Tensor):
        # print(x.size())
        h = self.norm1(x)
        h = self.act(h)                     # b, c, h, w
        h = self.conv1(h)

        time_emb = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)   # b, c -> b, c, 1, 1
        scale, shift = torch.chunk(time_emb, 2, dim=1)
        h = h * (scale + 1) + shift
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.skip_connect(x)


class SelfAttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 norm_type="batchnorm",):
        super(SelfAttnBlock, self).__init__()
        self.norm = create_norm(dim, norm_type)
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.k = nn.Conv2d(dim, dim, kernel_size=1)
        self.v = nn.Conv2d(dim, dim, kernel_size=1)
        self.out = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, x):
        
        n, c, h, w = x.shape
        norm_x = self.norm(x)
        q = self.q(norm_x)
        k = self.k(norm_x)
        v = self.v(norm_x)
        
        # n, c, h, w -> n, h*w, c
        q = q.reshape(n, c, h*w).permute(0, 2, 1)
        
        # n c h w -> n c h*w
        k = k.reshape(n, c, h*w)
        
        qk = torch.matmul(q, k)/math.sqrt(c)
        qk = F.softmax(qk, dim=-1)
        # qk: n, h*w, h*w
        
        v = v.reshape(n, c, h*w).permute(0, 2, 1)
        res = torch.bmm(qk, v)
        res = res.reshape(n, c, h, w)
        res = self.out(res)
        
        return x + res
        
        


class ResAttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_channels,
                 with_attn=False,
                 norm_type="batchnorm",
                 activation="lrelu"):
        super(ResAttnBlock, self).__init__()
        self.res_block = ResBlock(in_channels, out_channels, time_emb_channels, norm_type=norm_type, activation=activation)
        if with_attn:
            self.attn_block = SelfAttnBlock(out_channels, norm_type=norm_type)
        else:
            self.attn_block = nn.Identity()
            
    def forward(self, x, t):
        x = self.res_block(x, t)
        x = self.attn_block(x)
        return x


class ResAttnBlockMiddle(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_channels,
                 with_attn=False,
                 norm_type="batchnorm",
                 activation="lrelu"):
        super(ResAttnBlockMiddle, self).__init__()
        self.res_block1 = ResBlock(in_channels,
                                  out_channels,
                                  time_emb_channels,
                                  norm_type=norm_type,
                                  activation=activation)
        self.res_block2 = ResBlock(out_channels,
                                  out_channels,
                                  time_emb_channels,
                                  norm_type=norm_type,
                                  activation=activation)
        if with_attn:
            self.attn_block = SelfAttnBlock(out_channels, norm_type=norm_type)
        else:
            self.attn_block = nn.Identity()
            
    def forward(self, x, t):
        x = self.res_block1(x, t)
        x = self.attn_block(x)
        x = self.res_block2(x, t)
        return x


# upsample/downsample
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv=False, scale_factor=2):
        super(Upsample, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.use_conv = use_conv
        
        out_channels = default(out_channels, in_channels)
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        
    def forward(self, x: torch.Tensor):
        x = x.float()
        if self.use_conv:
            return self.conv(self.up(x))
        else:
            return self.up(x)
        
        

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv=False):
        super(Downsample, self).__init__()
        out_channels = default(out_channels, in_channels)
        if use_conv:
            self.op = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.op = nn.MaxPool2d(2, 2)
            
    def forward(self, x: torch.Tensor):
        return self.op(x)


# UNet Level
class UNetLevel(nn.Module):
    def __init__(self,
                 blocks: int,
                 inout_channels: int,
                 mid_channels: int,
                 time_emb_channels: int,
                 mid_block: nn.Module,
                 with_attn: bool = False,
                 norm_type="batchnorm",
                 activation="lrelu",
                 down_up_sample: bool = True
                 ):
        super(UNetLevel, self).__init__()
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        
        for _ in range(blocks):
            self.downs.append(ResAttnBlock(inout_channels,
                                           mid_channels,
                                           time_emb_channels,
                                           with_attn=with_attn,
                                           norm_type=norm_type,
                                           activation=activation))
            self.ups.insert(0, ResAttnBlock(mid_channels*2,
                                            inout_channels,
                                            time_emb_channels,
                                            with_attn=with_attn,
                                            norm_type=norm_type,
                                            activation=activation))
            inout_channels = mid_channels
        
        if down_up_sample:
            self.down = Downsample(mid_channels)
            self.up = Upsample(mid_channels)
        else:
            self.down = nn.Identity()
            self.up = nn.Identity()
        
        self.mid_block = mid_block
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        hs = []
        for down in self.downs:
            x = down(x, t_emb)
            hs.append(x)

        x = self.down(x)
        x = self.mid_block(x, t_emb)
        x = self.up(x)
        
        for up in self.ups:
            h = hs.pop()
            x = up(torch.concat((h,x), dim=1), t_emb)
        
        return x


class UNet(nn.Module):
    def __init__(
        self,
        blocks:int,
        img_channels:int,
        base_channels:int = 64,
        ch_mult: list = [1,2,4,4],
        pe_dim = 10,
        n_steps = 1000,
        norm_type="batchnorm",
        activation="lrelu",
        with_attn: Union[bool, List[bool]] = False,
        down_up_sample: bool = True
    ):
        super(UNet, self).__init__()
        
        # dims = [base_channels, base_channels, base_channels*2, base_channels*4, base_channels*4]
        self.levels = len(ch_mult)
        dims = [base_channels] + [int(base_channels * mult) for mult in ch_mult]
        self.with_attn = [with_attn] * self.levels if isinstance(with_attn, bool) else with_attn
        
        in_out = list(zip(dims[:-1], dims[1:], self.with_attn[:-1]))
        self.img_channels = img_channels
        
        
        
        self.in_proj = nn.Conv2d(img_channels, base_channels, 3, padding=1, stride=1)
        self.out_proj = nn.Sequential(
            # nn.GroupNorm(32, base_channels),
            # nn.Mish(),
            nn.BatchNorm2d(base_channels),
            nn.Conv2d(base_channels, img_channels, 3, padding=1, stride=1)
        )
        
        self.time_emb_dim = base_channels * 4
        # self.time_emb = TimeEmbedding(self.time_emb_dim)
        
        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.pe_linears = nn.Sequential(
            nn.Linear(pe_dim, self.time_emb_dim),
            create_activation(activation),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        
        # build unet
        # begin with middle block
        if self.with_attn[-1]:
            now_blocks = ResAttnBlockMiddle(base_channels*ch_mult[-1],
                                            base_channels*ch_mult[-1],
                                            self.time_emb_dim,
                                            with_attn=self.with_attn[-1],
                                            norm_type=norm_type,
                                            activation=activation)
        else:
            now_blocks = ResBlock(base_channels*ch_mult[-1], base_channels*ch_mult[-1], self.time_emb_dim)
        for inout_ch, mid_ch, attn in reversed(in_out):
            now_blocks = UNetLevel(blocks,
                                   inout_ch,
                                   mid_ch,
                                   self.time_emb_dim,
                                   mid_block=now_blocks,
                                   with_attn=attn,
                                   norm_type=norm_type,
                                   activation=activation,
                                   down_up_sample=down_up_sample)
        
        self.unet = now_blocks
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # t_emb = self.time_emb(t)
        t = self.pe(t)
        pe = self.pe_linears(t)
        
        x = self.in_proj(x)
        return self.out_proj(self.unet(x, pe))


if __name__ == "__main__":
    time_emb = TimeEmbedding(64)
    resblock = ResBlock(128, 128, 64)
    unet_level = UNetLevel(3, 64, 128, 64, resblock)
    selfattnblock = SelfAttnBlock(64)
    
    test_x = torch.randn(2,64,16,16)
    test_t = torch.rand((2))
    
    print('testing attn block')
    a = selfattnblock(test_x)
    print('a: ', a.shape)
    
    t_emb = time_emb(test_t)
    
    test_out = unet_level(test_x, t_emb)
    print(test_out.shape)
    print()
    
    unet_config = {
        'blocks': 2,
        'img_channels': 1,
        'base_channels': 4,
        'ch_mult': [1,2,4,4],
        'norm_type': 'batchnorm',
        'activation': 'lrelu',
        'with_attn': [False,False,False,True],
        'down_up_sample': False
    }
    
    unet = UNet(**unet_config).to('cuda')
    test_out = unet(test_x, test_t)
    print(f'hahaha: {torch.cuda.memory_allocated()}')
    print(test_out.shape)
    