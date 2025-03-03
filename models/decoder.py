from torch import nn
from typing import List, Tuple
from torch.nn import functional as F
import torch
import numpy as np

from .basicblocks import ResBlock, Upsample, make_attn

class Decoder(nn.Module):
    def __init__(self, out_channels:int, middle_channels: int, temb_ch: int=0, 
                 ch_mult:Tuple[int] = (1, 2, 4, 8), n_res_blocks:int = 2, 
                 attn_resolutions:List[int] = [16, 8], dropout:float=0.0, resamp_with_conv=True, 
                 resolution=256, z_channels=64, user_linear_attn=False, attn_type='vanilla',
                 tanh_out:bool=False, give_pre_end: bool=False):
        
        super().__init__()

        if user_linear_attn: attn_type = 'linear'
        self.middle_channels = middle_channels
        self.n_resolutions = len(ch_mult)
        self.n_res_blocks = n_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = middle_channels*ch_mult[self.n_resolutions-1]
        curr_res = resolution // 2**(self.n_resolutions-1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)

        self.mid.block_2 = ResBlock(in_channels=block_in,
                                    out_channels=block_in,
                                    temb_channels=temb_ch,
                                    dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.n_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = middle_channels * ch_mult[i_level]
            for _ in range(self.n_res_blocks+1):
                block.append(ResBlock(in_channels=block_in,
                                      out_channels=block_out,
                                      temb_channels=temb_ch,
                                      dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, temb=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.n_resolutions)):
            for i_block in range(self.n_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h        