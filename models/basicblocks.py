from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=32):
        super(ResBlock, self).__init__()
        self.groupnorm_1 = nn.GroupNorm(n_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(n_groups, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        residue = self.residual_layer(x)
        x = F.silu(self.groupnorm_1(x))
        x = self.conv_1(x)
        x = F.silu(self.groupnorm_2(x))
        x = self.conv_2(x)
        return x + residue