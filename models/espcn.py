import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * scale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        return self.model(x)
