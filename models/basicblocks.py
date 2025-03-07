from torch import nn
from torch.nn import functional as F
import torch
import math
from einops import rearrange
from abc import abstractmethod

from .utils import zero_module

def make_attn(block_in, attn_type):
    if attn_type == 'vanilla':
        return SpatialAttention(block_in)
    elif attn_type == 'linear':
        return LinearAttention(block_in)
    else:
        raise NotImplemented("You must choose 'vanilla' or 'linear' attention block.")

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimeStepResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 temb_channels: int, dropout: float=0.0, 
                 n_groups=32, eps:float=1e-6):
        
        super(ResBlock, self).__init__()
        self.groupnorm_1 = nn.GroupNorm(n_groups, in_channels, eps=eps)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(n_groups, out_channels, eps=eps)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        if temb_channels > 0:
            self.temb_layer = nn.Linear(temb_channels, out_channels)
        self.silu = nn.SiLU()

    def forward(self, x, temb=None):
        residue = self.residual_layer(x)

        x = self.silu(self.groupnorm_1(x))
        x = self.conv_1(x)
        
        if temb is not None:
            # Auto Casted
            x = x + self.temb_layer(self.silu(x))[:, :, None, None]

        x = self.silu(self.groupnorm_2(x))
        x = self.dropout(x)
        x = self.conv_2(x)
        return x + residue
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, dropout:float=0.0, dim_head:int=64, n_heads: int=4):
        super().__init__()
        self.q_embed = nn.Linear(query_dim, dim_head * n_heads, bias=False) 
        self.kv_embed = nn.Linear(context_dim, 2 * dim_head * n_heads, bias=False)
        self.out_proj = nn.Linear(dim_head * n_heads, query_dim)
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.constant = 1 / math.sqrt(dim_head)

    def forward(self, x: torch.Tensor, context=None):
        # context is the text  
        b, x_len, dim_q = x.shape
        
        if context is not None:
            if context.ndim == 4:
                # Image Prompt
                b, c, h, w = context.shape
                context = context.view(b, c, h*w).transpose(1, 2)
        else:
            context = x
        
        q = self.q_embed(x)
        # k, v shapes : (b, s, n_head * dim_head)
        k, v = torch.chunk(self.kv_embed(context), 2, dim=-1)

        q = q.view(b, -1, self.n_heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.n_heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.n_heads, self.dim_head).transpose(1, 2)

        # sim shape : (b, n_head, x_len, dim_head) X (b, n_head, dim_head, con_len) 
        # -> (b, n_head, x_len, con_len)
        sim = q @ k.transpose(-1, -2)
        sim *= self.constant

        sim = F.softmax(sim, dim=-1)
        # (b, n_head, x_len, con_len) X (b, n_head, con_len, dim_head) 
        # -> (b, n_head, x_len, dim_head)
        output = sim @ v
        output = output.contiguous().transpose(1, 2).reshape(b, x_len, -1)
        return self.dropout(self.out_proj(output))

class SpatialAttention(nn.Module):
    # Single Head
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.qkv_embed = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1)
        self.normalize = nn.GroupNorm(32, in_channels, eps=eps)
        self.constant = 1.0 / math.sqrt(in_channels)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, causal_mask=None):
        residue = x
        b, c, h, w = x.shape
        x = self.normalize(x)

        # q, k, v shapes : (b, c, h, w)
        q, k, v = torch.chunk(self.qkv_embed(x), chunks=3, dim=1)
        
        q = q.contiguous().view(b, c, h*w)
        k = k.contiguous().view(b, c, h*w)
        v = v.contiguous().view(b, c, h*w)

        # (b, hw, hw)
        attn = k.transpose(1, 2) @ q
        
        if causal_mask is not None:
            # Since we are not using auto-regressive model, this will be skipped in general learning pipelines.
            mask = torch.ones_like(attn, dtype=torch.bool).triu(1)
            # Upper triangular components are set to -torch.inf
            attn.masked_fill_(mask, -torch.inf)
            
        attn *= self.constant
        soft_attn = F.softmax(attn, dim=2)
        output = (v @ soft_attn).view(b, c, h, w)

        return residue + self.proj_out(output)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, n_heads, dim_head, dropout:float=0.0, use_conv: bool=False):
        super().__init__()
        assert in_channels == n_heads * dim_head
        self.qkv_embeds = nn.Linear(in_channels, 3 * n_heads * dim_head) if not use_conv else nn.Conv1d(in_channels, 3 * n_heads, * dim_head, 1)
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.constant = 1.0 / math.sqrt(dim_head)
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        
        # feedforward
        # I don't know why Stable Diffusion authors apply the dropout in here.
        self.ff = nn.Sequential(*[nn.Linear(n_heads * dim_head, in_channels),
                                nn.Dropout(dropout)]) if not use_conv else zero_module(nn.Conv1d(n_heads * dim_head, in_channels, 1))
        
    def forward(self, x, causal_mask=None):
        # x : (batch_size, sequence_length, feature_dim)
        residue = x
        x = self.norm(x.transpose(1, 2)).transpose(1,2)
        
        bsz, s_len, x_dim = x.shape
        # q, k, v : (bsz, s_len, n_heads * dim_head)
        q, k ,v = torch.chunk(self.qkv_embeds(x), 3, dim=-1)

        q = q.contiguous().view(bsz, s_len, self.n_heads, self.dim_head).transpose(1, 2)
        k = k.contiguous().view(bsz, s_len, self.n_heads, self.dim_head).transpose(1, 2)
        # bsz, n_heads, s_len, dim_head
        v = v.contiguous().view(bsz, s_len, self.n_heads, self.dim_head).transpose(1, 2)
        
        # bsz, n_heads, s_len, s_len
        attn = q @ k.transpose(-1, -2)
        attn *= self.constant

        if causal_mask is not None:
            # Since we are not using auto-regressive model, this will be skipped in general learning pipelines.
            mask = torch.ones_like(attn, dtype=torch.bool).triu(1)
            # Upper triangular components are set to -torch.inf
            attn.masked_fill_(mask, -torch.inf)

        attn = F.softmax(attn, dim=-1)
        # (bsz, n_heads, s_len, s_len) X (bsz, n_heads, s_len, dim_head) -> (bsz, n_heads, s_len, dim_head)
        out = attn @ v 
        out = out.transpose(1, 2).contiguous().view(bsz, s_len, self.n_heads * self.dim_head)
        return residue + self.ff(out)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv: bool=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.conv1(x)
        return self.conv2(F.silu(output))

class WaveletDownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.register_filters(in_channels)
        self.ffn = ConvFFN(in_channels, in_channels)
        self.in_channels = in_channels
        
    def register_filters(self, in_channels):
        low_filter = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
        high_filter = torch.tensor([1.0, -1.0]) / torch.sqrt(torch.tensor(2.0))
        LL = torch.outer(low_filter, low_filter)
        LH = torch.outer(low_filter, high_filter)
        HL = torch.outer(high_filter, low_filter)
        HH = torch.outer(high_filter, high_filter)
        dwt_filter = torch.stack([LL, LH, HL, HH]).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('dwt_filter', dwt_filter)
    
    def forward(self, x):
        b, c, h, w = x.shape
        avg_feat = F.adaptive_avg_pool2d(x, (h//2, w//2))
        gate_weights = F.sigmoid(self.ffn(avg_feat))
        y = F.conv2d(x, self.dwt_filter, stride=2, groups=self.in_channels)
        y = y.view(b, c, 4, h // 2, w // 2)
        LL, LH, HL, HH = y[:, :, 0, ...], y[:, :, 1, ...], y[:, :, 2, ...], y[:, :, 3, ...]
        output = gate_weights * LL + gate_weights * LH + gate_weights * HL + gate_weights * HH
        return output

class WaveletUpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.register_filters(in_channels)
        # Since there is no clear explanation how to chunk the input features, I chunck the feature arbitrary way.
        # When I think about the wavelet transform perspective, it is natural to use spatial kernel to get high pass and low pass results.
        # Otherwise, each chunk can be the identity of x
        self.splitter = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, bias=False, padding=1)
        self.ffn = ConvFFN(in_channels, in_channels)
        self.in_channels = in_channels
        
    def register_filters(self, in_channels):
        low_filter = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
        high_filter = torch.tensor([1.0, -1.0]) / torch.sqrt(torch.tensor(2.0))
        LL = torch.outer(low_filter, low_filter)
        LH = torch.outer(low_filter, high_filter)
        HL = torch.outer(high_filter, low_filter)
        HH = torch.outer(high_filter, high_filter)
        dwt_filter = torch.stack([LL, LH, HL, HH]).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('dwt_filter', dwt_filter)
    
    def forward(self, x: torch.Tensor):
        # The input feature is splitted into 4 chunks as the wavelet coefficients.
        b, c, h, w = x.shape
        LL, LH, HL, HH = torch.chunk(self.splitter(x), 4, dim=1)
        gate_weights = F.sigmoid(self.ffn(x))
        x = torch.cat([gate_weights * LL, gate_weights * LH, gate_weights * HL, gate_weights * HH], dim=1)
        y = F.conv_transpose2d(x, self.dwt_filter, stride=2, groups=c)
        return y
      
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv:bool=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(dim, mult * dim * 2)
        self.out = nn.Linear(mult * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = torch.chunk(self.proj(x), 2, dim=-1)
        x = x * F.gelu(gate)
        x = self.dropout(x)
        return self.out(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_head: int, dropout: float=0.0, context_dim: int=None,):
        super().__init__()
        self.msa = MultiHeadSelfAttention(in_channels=dim, n_heads=n_heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.cross_attn = CrossAttention(query_dim=dim, context_dim=context_dim, 
                                         n_heads=n_heads, dim_head=dim_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x, context):
        
        x = self.msa(self.norm1(x)) + x
        x = self.cross_attn(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, n_heads: int, dim_head: int, dropout:float=0.0, depth: int=1, context_dim:int=128):
        super().__init__()

        self.proj_in = nn.Conv2d(in_channels, n_heads * dim_head, kernel_size=1)
        self.norm = nn.GroupNorm(32, in_channels)
        self.former_blocks = nn.ModuleList([BasicTransformerBlock(n_heads * dim_head, n_heads, dim_head, dropout, context_dim) for _ in range(depth)])
        self.proj_out = zero_module(nn.Conv2d(n_heads * dim_head, in_channels, kernel_size=(1, 1)))
        self.n_heads = n_heads
        self.dim_head = dim_head

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        residue = x
        x = self.norm(x)
        # b, n_heads * dim_head, h, w
        x = self.proj_in(x)
        # b, 
        x = x.view(b, self.n_heads * self.dim_head, h * w).transpose(1, 2)
        
        for block in self.former_blocks:
            x = block(x, context)

        x = x.transpose(1, 2).view(b, self.n_heads * self.dim_head, h, w)
        x = self.proj_out(x)
        return x + residue

class TimeStepResBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, emb_channels, dropout:float=0.0, 
                 up: bool=False, down: bool=False, use_conv:bool=False, eps=1e-6):
        super().__init__()

        self.in_layers = nn.Sequential(*[nn.GroupNorm(32, in_channels, eps=eps),
                                         nn.SiLU(),
                                         nn.Conv2d(in_channels, out_channels, 3, padding=1)])
    
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, False)
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        self.emb_layers = nn.Sequential(nn.SiLU(),
                                        nn.Linear(emb_channels, out_channels))
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels, eps=eps),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )        

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:        
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, emb):

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)

        return self.skip_connection(x) + h
            
if __name__ == "__main__":
    msa = MultiHeadSelfAttention(256, 4, 64)
    sample_tensor = torch.randn((4, 16, 256))
    msa(sample_tensor)