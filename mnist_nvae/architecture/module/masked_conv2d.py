import numpy as np
import torch
from torch import nn


# TODO: refactor this
class MaskedConv2d(nn.Conv2d):
    
    def __init__(self, *args, mask_type='B', **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        # Mask A) without center pixel
        # Mask B) with center pixel

        # 1 1 1 1 1 1 1
        # 1 1 1 1 1 1 1
        # 1 1 1 X 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0

        mask = torch.ones_like(self.weight)
        _, _, height, width = self.weight.size()
        
        mask[:, :, height // 2, width // 2 + (1 if mask_type == 'B' else 0):] = 0
        mask[:, :, height // 2 + 1:] = 0

        self.mask = nn.Parameter(mask, requires_grad=False)
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
