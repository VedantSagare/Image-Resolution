import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class EDSR(nn.Module):
    def __init__(self, scale_factor=2, blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, 64, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(64, 3 * scale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)
