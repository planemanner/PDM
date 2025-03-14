import torch
from torch import nn
from torch.nn import functional as F
from .basicblocks import Downsample, Upsample, SpatialTransformer, MultiHeadSelfAttention, TimestepEmbedSequential, TimeStepResBlock
from .utils import TimeStep2Sinusoid, zero_module

class UNetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # | ----- Time Step Part ----- | #
        time_embed_dim = cfg.middle_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        """
        step_embed 는 time_embed_dim 과 같은 크기의 tensor 를 받도록 구성
        출력도 마찬가지.
        """
        self.use_step_embed = cfg.use_step_embed
        if self.use_step_embed:
            self.step_embed = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        self.get_sinusoid_time = TimeStep2Sinusoid(time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(cfg.in_channels, cfg.middle_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = cfg.middle_channels
        input_block_chans = [cfg.middle_channels]
        ch = cfg.middle_channels

        ds = 1

        for level, mult in enumerate(cfg.ch_mult):
            for _ in range(cfg.n_res_blocks):
                layers = [
                    TimeStepResBlock(ch, 
                                     mult * cfg.middle_channels, 
                                     t_emb_channels=time_embed_dim, 
                                     d_emb_channels=time_embed_dim,
                                     dropout=cfg.dropout)
                ]
                ch = mult * cfg.middle_channels
                if ds in cfg.attn_resolutions:
                    if cfg.num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // cfg.num_head_channels
                        dim_head = cfg.num_head_channels

                    dim_head = ch // num_heads if cfg.use_spatial_transformer else cfg.num_head_channels
                    layers.append(
                        MultiHeadSelfAttention(
                            ch,
                            n_heads=num_heads,
                            dim_head=dim_head,
                            use_conv=True
                        ) if not cfg.use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=cfg.transformer_depth, context_dim=cfg.context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(cfg.ch_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(

                        TimeStepResBlock(in_channels=ch, 
                                         out_channels=out_ch, 
                                         t_emb_channels=time_embed_dim,
                                         d_emb_channels=time_embed_dim,
                                         down=True) if cfg.resblock_updown else Downsample(ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if cfg.num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // cfg.num_head_channels
            dim_head = cfg.num_head_channels
    
        dim_head = ch // num_heads if cfg.use_spatial_transformer else cfg.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            TimeStepResBlock(ch, 
                             ch, 
                             t_emb_channels=time_embed_dim,
                             d_emb_channels=time_embed_dim,
                             dropout=cfg.dropout),

            MultiHeadSelfAttention(ch,
                                   n_heads=num_heads, 
                                   dim_head=dim_head, 
                                   use_conv=True) if not cfg.use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=cfg.transformer_depth, context_dim=cfg.context_dim
                        ),
            TimeStepResBlock(ch, 
                             ch, 
                             t_emb_channels=time_embed_dim,
                             d_emb_channels=time_embed_dim,
                             dropout=cfg.dropout),
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(cfg.ch_mult))[::-1]:
            for i in range(cfg.n_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    TimeStepResBlock(
                        ch + ich,
                        cfg.middle_channels * mult,
                        t_emb_channels=time_embed_dim,
                        d_emb_channels=time_embed_dim,
                        dropout=cfg.dropout)                    
                ]
                ch = cfg.middle_channels * mult
                if ds in cfg.attn_resolutions:
                    if cfg.num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // cfg.num_head_channels
                        dim_head = cfg.num_head_channels
                    
                    dim_head = ch // num_heads if cfg.use_spatial_transformer else cfg.num_head_channels

                    layers.append(
                        MultiHeadSelfAttention(
                            ch,
                            n_heads=num_heads,
                            dim_head=dim_head,
                            use_conv=True,
                        ) if not cfg.use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=cfg.transformer_depth, context_dim=cfg.context_dim
                        )
                    )
                if level and i == cfg.n_res_blocks:
                    out_ch = ch
                    layers.append(
                        TimeStepResBlock(
                            ch,
                            out_ch,
                            t_emb_channels=time_embed_dim,
                            d_emb_channels=time_embed_dim,
                            dropout=cfg.dropout,
                            up=True
                        )                        
                        if cfg.resblock_updown else Upsample(ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(cfg.middle_channels, cfg.out_channels, 3, padding=1))
        )

    def forward(self, x: torch.FloatTensor, timesteps: torch.IntTensor, step_sizes: torch.IntTensor=None, context: torch.FloatTensor=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        # In this repo, I removed the label part for simplicity.
        """
        hs = []
        t_emb = self.get_sinusoid_time(timesteps)
        # sinusoidal embedding 이 적절한지 여부에 대해 고민해볼 필요 있음.
        t_emb = self.time_embed(t_emb)

        if step_sizes is not None and self.use_step_embed:
            d_emb = self.get_sinusoid_time(step_sizes)
            d_emb = self.step_embed(d_emb)
        else:
            # Flow Matching 안 쓸 때 분기 처리
            d_emb = None

        h = x

        for i, module in enumerate(self.input_blocks):
            
            h = module(h, t_emb, d_emb, context)
            hs.append(h)

        h = self.middle_block(h, t_emb, d_emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, t_emb, d_emb, context)
            
        return self.out(h)

            
if __name__ == "__main__":
    a = nn.Sequential(*[nn.Linear(3, 4),
                        nn.Linear(4, 4)])
    class DotDict(dict):
        def __init__(self, d: dict={}):
            super().__init__()
            for key, value in d.items():
                self[key] = DotDict(value) if type(value) is dict else value

        def __getattr__(self, key):
            if key in self:
                return self[key]
            raise AttributeError(key)
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__    

    unet_cfg = {"middle_channels":128,
            "in_channels": 3, # color
            "out_channels": 3,
            "ch_mult": (1, 1, 2, 2, 4, 4),
            "use_timestep": True,
            "resolution": 256,
            "n_res_blocks": 2,
            "dropout": 0.0,
            "attn_type": 'vanilla',
            "attn_resolutions": [16, 8],
            "resamp_with_conv": True,
            "transformer_depth": 1,
            "context_dim": 128,
            "use_spatial_transformer": True,
            "num_head_channels": 32,
            "resblock_updown": False}
    
    unet_cfg = DotDict(unet_cfg)
    unet = UNetModel(unet_cfg)

    bsz, c, h, w = 4, 3, 32, 32
    tkn_len = 77
    sample_images = torch.randn((bsz, c, h, w))
    sample_time_steps = torch.randint(1, 1000, (bsz,))
    sample_context = torch.randn((bsz, tkn_len, unet_cfg.context_dim))
    eps = unet(sample_images, sample_time_steps, sample_context)
    print(eps.shape)