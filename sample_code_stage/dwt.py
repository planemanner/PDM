import torch
import torch.nn.functional as F

class HaarWaveletTransform2D:
    def __init__(self):
        # Haar 1D filters
        self.low_filter = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
        self.high_filter = torch.tensor([1.0, -1.0]) / torch.sqrt(torch.tensor(2.0))
        
        # Create 2D filters via outer product
        self.register_filters()

    def register_filters(self):
        # 2D filters for LL, LH, HL, HH
        LL = torch.outer(self.low_filter, self.low_filter)
        LH = torch.outer(self.low_filter, self.high_filter)
        HL = torch.outer(self.high_filter, self.low_filter)
        HH = torch.outer(self.high_filter, self.high_filter)

        # Shape: (out_channels=4, in_channels=1, kernel_height, kernel_width)
        self.dwt_filters = torch.stack([LL, LH, HL, HH]).unsqueeze(1)  # (4, 1, 2, 2)
        self.idwt_filters = self.dwt_filters  # Same filters for inverse

    def dwt(self, x):
        """
        Discrete Wavelet Transform (DWT)
        Args:
            x: Input tensor with shape (b, c, h, w)
        Returns:
            LL, LH, HL, HH: Sub-bands with shape (b, c, h//2, w//2)
        """
        b, c, h, w = x.shape
        filters = self.dwt_filters.to(x.device).repeat(c, 1, 1, 1)  # (4*c, 1, 2, 2)
        
        # Apply filters using grouped convolution
        y = F.conv2d(x, filters, stride=2, groups=c)  # (b, 4*c, h//2, w//2)
        y = y.view(b, c, 4, h // 2, w // 2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return LL, LH, HL, HH

    def idwt(self, LL, LH, HL, HH):
        """
        Inverse Discrete Wavelet Transform (IDWT)
        Args:
            LL, LH, HL, HH: Sub-bands with shape (b, c, h, w)
        Returns:
            Reconstructed tensor with shape (b, c, h*2, w*2)
        """
        b, c, h, w = LL.shape
        filters = self.idwt_filters.to(LL.device).repeat(c, 1, 1, 1)  # (4*c, 1, 2, 2)

        # Stack sub-bands into a single tensor
        y = torch.stack([LL, LH, HL, HH], dim=2).view(b, c * 4, h, w)
        
        # Transposed convolution (upsampling + filtering)
        x_reconstructed = F.conv_transpose2d(y, filters, stride=2, groups=c)  # (b, c, h*2, w*2)
        return x_reconstructed

if __name__ == "__main__":
    # Initialize transform
    # wavelet = HaarWaveletTransform2D()

    # # Dummy input tensor (batch=2, channels=3, height=64, width=64)
    # x = torch.randn(2, 3, 64, 64)

    # # Perform DWT
    # LL, LH, HL, HH = wavelet.dwt(x)
    # print(f"LL shape: {LL.shape}, LH shape: {LH.shape}, HL shape: {HL.shape}, HH shape: {HH.shape}")

    # # Perform IDWT
    # x_reconstructed = wavelet.idwt(LL, LH, HL, HH)
    # print(f"Reconstructed shape: {x_reconstructed.shape}")

    # # Check reconstruction error
    # reconstruction_error = torch.norm(x - x_reconstructed) / torch.norm(x)
    # print(f"Reconstruction error: {reconstruction_error:.6f}")
    x = [i for i in range(100)]
    print(x[::2])