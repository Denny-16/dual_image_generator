# dual_image_rams.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Multi-Scale Feature Extractor -----
class MultiScaleExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, 32, kernel_size=7, padding=3)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.conv7x7(x)
        return torch.cat([x1, x2, x3], dim=1)  # Output: [B, 96, H, W]


# ----- Channel Attention Block -----
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


# ----- ConvLSTM Cell (simplified) -----
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ----- Dual-Image RAMSNet -----
class DualImageRAMSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = MultiScaleExtractor()
        self.convlstm = ConvLSTMCell(input_dim=96, hidden_dim=64, kernel_size=3)
        self.attn = ChannelAttention(64)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)  # [B, 96, H, W]
        f2 = self.feature_extractor(x2)  # [B, 96, H, W]

        # Initialize hidden states
        h, c = torch.zeros_like(f1[:, :64]), torch.zeros_like(f1[:, :64])

        # Step 1
        h, c = self.convlstm(f1, h, c)
        # Step 2
        h, c = self.convlstm(f2, h, c)

        h = self.attn(h)
        out = self.reconstruct(h)  # [B, 3, H, W]
        return out


# ----- Example usage -----
if __name__ == "__main__":
    model = DualImageRAMSNet()
    lr1 = torch.randn(1, 3, 128, 128)  # Simulated input 1
    lr2 = torch.randn(1, 3, 128, 128)  # Simulated input 2
    out = model(lr1, lr2)
    print("Output shape:", out.shape)  # Should be [1, 3, 128, 128]
