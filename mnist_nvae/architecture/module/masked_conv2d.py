# %%
import numpy as np
import torch
from torch import nn


# TODO: refactor this
class MaskedConv2d(nn.Conv2d):
    
    def __init__(self, *args, mask_type='B', **kwargs):
        super().__init__(*args, **kwargs)
        
        # Mask A) without center pixel
        # Mask B) with center pixel

        # 1 1 1 1 1 1 1
        # 1 1 1 1 1 1 1
        # 1 1 1 X 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0

        mask = torch.ones_like(self.weight)
        _, _, height, width = self.weight.size()
        
        # out_channels, in_channels, h, w
        mask[:, :, height // 2, width // 2 + (1 if mask_type == 'B' else 0):] = 0
        mask[:, :, height // 2 + 1:] = 0

        self.mask = nn.Parameter(mask, requires_grad=False)
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
# %%

# conv = MaskedConv2d(3, 3, kernel_size=5, mask_type='A', bias=True, padding=2)
# # %%

# x = torch.ones(1, 3, 4, 4)
# # %%

# conv(x)[0,0]
# # %%

# conv(conv(x))[0,0]
# # %%

# conv(conv(conv(x)))[0,0]
# # %%

# conv(conv(conv(conv(x))))[0,0]
# # %%
# y = x
# for index in range(10):
#     print(index)
#     y2 = conv(y)
#     if index >= 1 and y[0,0].view(-1)[index - 1] != y2[0,0].view(-1)[index - 1]:
#         raise Exception('what??')
#     y = y2
#     print(y[0,0])
# # %%


# seq = nn.Sequential(
#     MaskedConv2d(3, 3, kernel_size=5, mask_type='A', bias=True, padding=2, unmasked_channels=2),
#     MaskedConv2d(3, 3, kernel_size=5, mask_type='B', bias=True, padding=2, unmasked_channels=2),
# )
# # %%
# print('start')
# x = torch.ones(1, 3, 4, 4)
# # %%

# seq(x)[0,-1]
# # %%

# seq(seq(x))[0,-1]
# # %%

# seq(seq(seq(x)))[0,0]
# # %%

# seq(seq(seq(seq(x))))[0,0]
# # %%
# y = x
# for index in range(10):
#     print(index)
#     y2 = seq(y)
#     if index >= 1 and y[0,0].view(-1)[index - 1] != y2[0,0].view(-1)[index - 1]:
#         raise Exception('what??')
#     y = y2
#     print(y[0,0])
# # %%