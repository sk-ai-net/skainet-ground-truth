import torch
from gt.core import Executable

"""
UC-003: Spatial map broadcasting (addition)

Covers:
- Operation: addition (+)
- Layout: NCHW
- Batch usage: yes (B > 1)
- Broadcasting pattern: (1,1,H,W) -> (B,C,H,W)
"""


@Executable("Addition with spatial map broadcasting (NCHW, batched)")
def add_spatial_map_broadcast_nchw_batched():
    # Input tensor in NCHW with batch
    x = torch.randn(2, 3, 6, 5, requires_grad=True)
    # Spatial map that broadcasts across batch and channels
    m = torch.randn(1, 1, 6, 5, requires_grad=True)
    y = x + m
    return [x, m], y
