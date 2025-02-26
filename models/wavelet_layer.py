import torch
from torch.nn import functional as F
from torch import nn

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

if __name__ == "__main__":
    waup = WaveletUpSample(128)
    wadown = WaveletDownSample(128)
    sample_input = torch.randn(4, 128, 16, 16)
    print(wadown(sample_input).shape)
        