import numpy as np
import torch
import torch.nn as nn

from mnist_nvae.architecture import module


class RandomFourier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError('Out channel must be divisible by 4')

        self.B = nn.Parameter(
            torch.randn((in_channels, out_channels // 2)), requires_grad=False
        )

    @staticmethod
    def gridspace(x):
        h, w = x.shape[-2:]
        grid_y, grid_x = torch.meshgrid([
            torch.linspace(0, 1, steps=h),
            torch.linspace(0, 1, steps=w)
        ])
        return (
            torch.stack([grid_y, grid_x])
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
            .to(x)
        )

    def forward(self, x):
        x_proj = ((2 * np.pi * x.transpose(1, -1)) @ self.B).transpose(1, -1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)


class FourierSampleDecoder(nn.Module):
    def __init__(self, fourier_channels, decoded):
        super().__init__()
        self.random_fourier = RandomFourier(2, fourier_channels)
        self.decoded = decoded

    def forward(self, sample):
        return self.decoded(
            torch.cat([
                self.random_fourier(module.RandomFourier.gridspace(sample)),
                sample,
            ], dim=1)
        )
