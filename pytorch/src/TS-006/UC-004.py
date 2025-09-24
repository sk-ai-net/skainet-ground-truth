import torch
from gt.core import Executable

"""
UC-004: Channel-wise bias broadcasting (subtraction)

Covers:
- Operation: subtraction (-)
- Layout: NCHW
- Batch usage: yes (B > 1)
- Broadcasting pattern: (1,C,1,1) -> (B,C,H,W)
"""


@Executable("Subtraction with channel-wise bias broadcasting (NCHW, batched)")
def sub_channel_bias_broadcast_nchw_batched():
    # Input tensor in NCHW with batch
    x = torch.randn(2, 5, 7, 7, requires_grad=True)
    # Per-channel bias that broadcasts across batch and spatial dims
    b = torch.randn(1, 5, 1, 1, requires_grad=True)
    y = x - b
    return [x, b], y
