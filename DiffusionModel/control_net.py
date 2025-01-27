import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .diffusion import DiffusionModel
from .networks import UNetLevel, Downsample, Upsample, SelfAttnBlock, ResAttnBlock

class zero_convolution(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(zero_convolution, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight.data.fill_(0)
        if bias:
            self.bias.data.fill_(0)


class Unet_Encoder(pl.LightningModule):
    def __init__(self, unet):  # 接收unet实例而非路径
        super(Unet_Encoder, self).__init__()
        self.encoder = torch.nn.ModuleList([])
        mid_block = unet.unet.mid_block
        self.encoder.append(unet.unet.downs)
        self.encoder.append(unet.unet.down)
        while isinstance(mid_block, UNetLevel):
            self.encoder.append(mid_block.downs)
            self.encoder.append(mid_block.down)
            mid_block = mid_block.mid_block
        self.encoder.append(mid_block)
        self.pe = unet.pe
        self.pe_proj = unet.pe_linears
        self.in_proj = unet.in_proj

    def forward(self, x, t):
        x = self.in_proj(x)
        t_enc = self.pe(t)
        t_emb = self.pe_proj(t_enc)
        control = []

        def apply_layers(layers, x, t_emb):
            for layer in layers:
                # print layer info
                # print(layer)
                if isinstance(layer, torch.nn.ModuleList):
                    # 如果是 ModuleList，递归调用
                    x = apply_layers(layer, x, t_emb)
                elif isinstance(layer, torch.nn.Conv2d):
                    x = layer(x)
                elif isinstance(layer, Downsample):
                    x = layer(x)
                else:
                    x = layer(x, t_emb)
            return x

        x = apply_layers(self.encoder, x, t_emb)
        return x

class Unet_Decoder(pl.LightningModule):
    def __init__(self, unet):  # 接收unet实例
        super(Unet_Decoder, self).__init__()
        self.decoder = torch.nn.ModuleList([])
        mid_block = unet.unet.mid_block
        self.decoder.insert(0, unet.unet.ups)
        self.decoder.insert(0, unet.unet.up)
        while isinstance(mid_block, UNetLevel):
            self.decoder.insert(0, mid_block.ups)
            self.decoder.insert(0, mid_block.up)
            mid_block = mid_block.mid_block
        self.out_proj = unet.out_proj
        self.pe = unet.pe
        self.pe_proj = unet.pe_linears
    
    def forward(self, x, t):
        
        # print(f'Input shape: {x.shape}')
        t_enc = self.pe(t)
        t_emb = self.pe_proj(t_enc)
        x = torch.cat((x, torch.zeros_like(x)), dim=1)
        def apply_layers(layers, x, t_emb):
            for idx, layer in enumerate(layers):
                # print(f'Layer: {layer.__class__.__name__}, Input shape: {x.shape}')
                if isinstance(layer, torch.nn.ModuleList):
                    x = apply_layers(layer, x, t_emb)
                elif isinstance(layer, Upsample):
                    x = layer(x)
                else:
                    x = layer(x, t_emb)
                if idx != (len(layers) - 1) and (not isinstance(layer, Upsample)):
                    x = torch.cat((x, torch.zeros_like(x)), dim=1)
                    # print(f'layer idx {idx}, After concat, Output shape: {x.shape}')
                # print(f'After {layer.__class__.__name__}, Output shape: {x.shape}')
                
            return x
        x = apply_layers(self.decoder, x, t_emb)
        return self.out_proj(x)


class Controlled_UNet_Level(pl.LightningModule):
    def __init__(self,
                 inout_channels,
                 mid_channels,
                 downs_block,
                 frozen_downs_block,
                 down_block,
                 frozen_down_block,
                 mid_block,
                 up_block,
                 frozen_ups_block):                                 
        super(Controlled_UNet_Level, self).__init__()
        self.frozen_downs = frozen_downs_block
        self.frozen_ups = frozen_ups_block
        # build control ups
        self.control_ups = torch.nn.ModuleList([])
        
        for layer in self.frozen_ups:
            if isinstance(layer, ResAttnBlock):
                self.control_ups.append(zero_convolution(layer.in_channels//2, layer.out_channels, 1, 1, 0))
            else:
                self.control_ups.append(torch.nn.Identity())
        # self.control_ups = torch.nn.ModuleList([zero_convolution(mid_channels, inout_channels*2, 1, 1, 0),
        #                                         zero_convolution(inout_channels*2, inout_channels, 1, 1, 0)])
        # TODO: Ups should be FREEZE!
        self.up = up_block.requires_grad_(False).eval()
        self.downs = downs_block
        self.down = down_block
        self.frozen_down = frozen_down_block
        self.mid_block = mid_block

    def forward(self, x, t_emb, xc):
        hs = []
        controls = []
        # calculate control
        for down in self.downs:
            xc = down(xc, t_emb)
            controls.append(xc)

        # pass through downsample layers and save features
        with torch.no_grad():
            for down in self.frozen_downs:
                x = down(x, t_emb)
                hs.append(x)

        xc = self.down(xc)
        x = self.frozen_down(x)
        x = self.mid_block(x, t_emb, xc)
        x = self.up(x)
        # print control_ups info
        # print(self.frozen_ups)
        # print(self.control_ups)
        
        # pass through upsample layers and fuse control
        for i, up in enumerate(self.frozen_ups):
            h = hs.pop()
            cont = controls.pop()
            with torch.no_grad():
                x = torch.cat((h,x), dim=1)
                x = up(x, t_emb)
            cont = self.control_ups[i](cont)
            # print(f'max value of cont: {torch.max(cont)}')
            # print(f'Control shape: {cont.shape}, Output shape: {x.shape}')
            x = x + cont
        
        return x

class Controlled_midblock(pl.LightningModule):
    def __init__(self,
                 trainable_mid_block,
                 frozen_mid_block,
                 mid_block_channels):
        super(Controlled_midblock, self).__init__()
        self.frozen_mid_block = frozen_mid_block.requires_grad_(False).eval()
        self.trainable_mid_block = trainable_mid_block
        self.control_mid = zero_convolution(mid_block_channels, mid_block_channels, 1, 1, 0)
    
    def forward(self, x, t_emb, xc):
        cont = self.control_mid(self.trainable_mid_block(xc, t_emb))
        # print(f'max value of cont: {torch.max(cont)}')
        x = self.frozen_mid_block(x, t_emb)
        return x + cont

class Controlled_UNet(pl.LightningModule):
    def __init__(self, 
                 diffusion_model_dir='./DiffusionModel/ckpts_celebHQ64_wo_condition/last.ckpt', 
                 map_location='cuda',
                 hint_channels=3):
        super(Controlled_UNet, self).__init__()
        
        # init trained diffusion model
        self.diffusion_model = DiffusionModel.load_from_checkpoint(diffusion_model_dir, map_location=map_location)
        origin_unet = self.diffusion_model.unet
        
        # build control net
        self.hint_in_proj = zero_convolution(hint_channels, origin_unet.base_channels, 1, 1, 0)
        self.x_in_proj = origin_unet.in_proj.requires_grad_(False).eval()
        self.x_out_proj = origin_unet.out_proj.requires_grad_(False).eval()
        self.in_out = origin_unet.in_out
        
        # start from mid_block
        frozen_unet_encoder = Unet_Encoder(origin_unet).encoder.requires_grad_(False).eval()
        trainable_unet_encoder = Unet_Encoder(origin_unet).encoder
        frozen_mid_block = frozen_unet_encoder.pop(-1)
        trainable_mid_block = trainable_unet_encoder.pop(-1)
        frozen_unet_decoder = Unet_Decoder(origin_unet).decoder.requires_grad_(False).eval()

        now_blocks = Controlled_midblock(trainable_mid_block, frozen_mid_block, self.diffusion_model.unet.base_channels*self.diffusion_model.unet.ch_mult[-1])
        # TODO: haven't checked the correctness of the following code 20250124
        for inout, mid, _ in reversed(self.diffusion_model.unet.in_out):
            trainable_down_block = trainable_unet_encoder.pop(-1)
            trainable_downs_block = trainable_unet_encoder.pop(-1)
            frozen_down_block = frozen_unet_encoder.pop(-1)
            frozen_downs_block = frozen_unet_encoder.pop(-1)
            frozen_up_block = frozen_unet_decoder.pop(0)
            frozen_ups_block = frozen_unet_decoder.pop(0)
            now_blocks = Controlled_UNet_Level(inout, mid, trainable_downs_block, frozen_downs_block, trainable_down_block,frozen_down_block, now_blocks, frozen_up_block, frozen_ups_block)

        self.controlnet = now_blocks

    def forward(self, x, t, hint):
        # project hint to the same dimension as x
        if (hint.shape[-1], hint.shape[-2]) != (x.shape[-1], x.shape[-2]):
            hint = F.interpolate(hint, (x.shape[-2], x.shape[-1]), mode='bilinear')
        hint = self.hint_in_proj(hint)
        x = self.x_in_proj(x)
        xc = x + hint


        t_enc = self.diffusion_model.unet.pe(t)
        t_emb = self.diffusion_model.unet.pe_linears(t_enc)
        # get encoder output
        x = self.controlnet(x, t_emb, xc)
        
        
        # add control output to the original output
        return self.x_out_proj(x)


if __name__ == '__main__':

    # DiffusionModel
    dm = DiffusionModel.load_from_checkpoint('/home/wty/mmf_-demo/DiffusionModel/ckpts_celebHQ64_wo_condition/last.ckpt', map_location='cuda')
    origin_unet = dm.unet

    copyed_encoder = Unet_Encoder(dm.unet).to('cuda')
    copyed_decoder = Unet_Decoder(dm.unet).to('cuda')
    # print(copyed_encoder.encoder)
    # print(copyed_decoder.decoder)


    bs = 5
    h = w = 64
    x = torch.ones(bs, 3, h, w).to('cuda')
    x_copy = x
    hint = torch.randn(bs, 3, h, w).to('cuda')
    print(torch.max(hint))
    t = [int(100)] * bs
    # send t to cuda
    t = torch.tensor(t).to('cuda')
    x = copyed_encoder(x, t)
    print(x.shape)

    x = copyed_decoder(x, t)
    print(x.shape)
    x_o = x

    x_in = torch.ones(bs, 3, h, w).to('cuda')
    controlled_unet = Controlled_UNet().to('cuda')
    with torch.no_grad():
        x_o = controlled_unet.diffusion_model.unet(x_in, t)
        x_copy = controlled_unet(x_in, t, hint)
    print(x_copy.shape)

    print(torch.max(x_copy - x_o))
    print(torch.min(x_copy - x_o))