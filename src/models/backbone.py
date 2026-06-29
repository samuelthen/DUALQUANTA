"""
NAFNet backbone blocks used by SPADNet.

SimpleGate, NAFBlock, DownBlock, UpBlock.
"""

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Split channels in half and multiply: output = a * b."""
    def forward(self, x):
        a, b = x.chunk(2, 1)
        return a * b


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free block from NAFNet.
    Uses SimpleGate instead of GELU/ReLU, LayerNorm for normalisation.
    """
    def __init__(self, c, dw=2, ff=2):
        super().__init__()
        dc = c * dw
        fc = c * ff
        self.n1  = nn.LayerNorm(c)
        self.c1  = nn.Conv2d(c, dc, 1)
        self.dw  = nn.Conv2d(dc, dc, 3, 1, 1, groups=dc)
        self.sg  = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dc // 2, dc // 2, 1))
        self.c2  = nn.Conv2d(dc // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.n2  = nn.LayerNorm(c)
        self.c3  = nn.Conv2d(c, fc, 1)
        self.sg2 = SimpleGate()
        self.c4  = nn.Conv2d(fc // 2, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        r = x
        x = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.sg(self.dw(self.c1(x)))
        x = x * self.sca(x)
        r = r + self.c2(x) * self.beta
        x = r
        x = self.n2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return r + self.c4(self.sg2(self.c3(x))) * self.gamma


class DownBlock(nn.Module):
    """NAFBlocks followed by PixelUnshuffle downsampling (2x)."""
    def __init__(self, ci, co, nb):
        super().__init__()
        self.blocks = nn.Sequential(*[NAFBlock(ci) for _ in range(nb)])
        self.down   = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(ci * 4, co, 3, 1, 1))

    def forward(self, x):
        x = self.blocks(x)
        return self.down(x), x


class UpBlock(nn.Module):
    """PixelShuffle upsampling (2x) followed by skip-add and NAFBlocks."""
    def __init__(self, ci, cs, co, nb):
        assert ci == 2 * co and cs == co
        super().__init__()
        self.up     = nn.Sequential(nn.Conv2d(ci, ci * 2, 1), nn.PixelShuffle(2))
        self.blocks = nn.Sequential(*[NAFBlock(co) for _ in range(nb)])

    def forward(self, x, skip):
        return self.blocks(self.up(x) + skip)
