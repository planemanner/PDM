from torch import nn
from basicblocks import ResBlock, Downsample, make_attn
from typing import List, Tuple
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels:int, middle_channels: int, temb_ch: int, 
                 ch_mult: Tuple[int]=(1, 2, 4, 8), n_res_blocks=2,
                 attn_resolutions:List[int]= [16, 8], dropout=0.0, 
                 resamp_with_conv=True, resolution=256, z_channels=64, 
                 double_z=True, user_linear_attn=False, attn_type='vanilla'):
        
        super().__init__()
        
        if user_linear_attn: attn_type = 'linear'

        self.first_conv = nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.n_resolutions = len(ch_mult)
        self.n_res_blocks = n_res_blocks
        
        curr_res = resolution

        for i in range(len(ch_mult)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = middle_channels * in_ch_mult[i]
            block_out = middle_channels * ch_mult[i]

            for _ in range(n_res_blocks):
                block.append(ResBlock(in_channels=block_in,
                                      out_channels=block_out,
                                      temb_channels=temb_ch,
                                      dropout=dropout))
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))    

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i != self.n_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(down)

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

        # end
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  2*z_channels if double_z else z_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x, temb=None):
        # timestep embedding

        # downsampling
        hs = [self.first_conv(x)]

        for i_level in range(self.n_resolutions):

            for i_block in range(self.n_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.n_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = F.sliu(h)
        h = self.conv_out(h)
        return h
                                                   
if __name__ == "__main__":
    pass
# ch_mult=(1, 2, 4, 8)
# in_ch_mult = (1,)+tuple(ch_mult)
# print(in_ch_mult)