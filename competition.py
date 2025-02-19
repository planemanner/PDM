
from torch import nn
from einops import rearrange, einsum
from inspect import isfunction
import math
import torch
from torch.nn import functional as F
from time import time

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttentionDefault(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, dropout:float=0.0, dim_head:int=64, n_head: int=4):
        super().__init__()
        self.q_embed = nn.Linear(query_dim, dim_head * n_head, bias=False) 
        self.kv_embed = nn.Linear(context_dim, 2 * dim_head * n_head, bias=False)
        self.out_proj = nn.Linear(dim_head * n_head, query_dim)
        self.n_head = n_head
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.constant = 1 / math.sqrt(dim_head)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # context is the text  
        b, x_len, dim_q = x.shape
        _, con_len, dim_kv = context.shape
        # print(x.transpose(1,2).shape)
        # print(self.q_embed.weight.shape)
        q = self.q_embed(x)
        # k, v shapes : (b, s, n_head * dim_head)
        k, v = torch.chunk(self.kv_embed(context), 2, dim=-1)

        q = q.view(b, -1, self.n_head, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.n_head, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.n_head, self.dim_head).transpose(1, 2)

        # sim shape : (b, n_head, x_len, dim_head) X (b, n_head, dim_head, con_len) 
        # -> (b, n_head, x_len, con_len)
        sim = q @ k.transpose(-1, -2)
        sim *= self.constant

        sim = F.softmax(sim, dim=-1)
        # (b, n_head, x_len, con_len) X (b, n_head, con_len, dim_head) 
        # -> (b, n_head, x_len, dim_head)
        output = sim @ v
        # (b, x_len, n_head, dim_head) -> (b, x_len, n_head * dim_head)
        output = output.contiguous().transpose(1, 2).reshape(b, x_len, -1)
        
        return self.dropout(self.out_proj(output))
    
if __name__ == "__main__":
    bsz = 8
    ids = torch.round(torch.linspace(0, 1000 - 1, 1)).long()
    ts = torch.full((bsz, ), 2)
    print(ids[ts])

    pass
    # q_dim, con_dim, n_heads, dim_head = 32, 64, 4, 64
    # bsz, q_len, c_len = 4, 16, 32
    # device = 'cuda:0'
    # cross_1 = CrossAttention(query_dim=q_dim, context_dim=con_dim, heads=n_heads, dim_head=dim_head)
    # cross_2 = CrossAttention2(query_dim=q_dim, context_dim=con_dim, dim_head=dim_head, n_head=n_heads)
    # cross_1 = cross_1.to(device)
    # cross_2 = cross_2.to(device)
    # sample_query = torch.randn(bsz, q_len, q_dim).to(device)
    # sample_context = torch.randn(bsz, c_len, con_dim).to(device)

    # time_1, time_2 = [], []
    # num_iters = 100

    # for i in range(num_iters):
    #     start_time = time()
    #     v1 = cross_1(sample_query, sample_context)
    #     elapsed_time1 = time() - start_time
    #     time_1.append(elapsed_time1)
    #     start_time = time()
    #     v2 = cross_2(sample_query, sample_context)
    #     elapsed_time2 = time() - start_time
    #     time_2.append(elapsed_time2)
    # import numpy as np
    # print(np.mean(time_1), np.mean(time_2))
        
