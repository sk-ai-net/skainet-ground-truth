import torch
from gt.core import Executable

"""
UC-002: Channel-wise bias broadcasting (addition)

Covers:
- Operation: addition (+)
- Layout: NCHW
- Batch usage: yes (B > 1)
- Broadcasting pattern: (1,C,1,1) -> (B,C,H,W)
"""


@Executable("Addition with channel-wise bias broadcasting (NCHW, batched)")
def add_channel_bias_broadcast_nchw_batched():
    # Input tensor in NCHW with batch
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    # Per-channel bias that broadcasts across batch and spatial dims
    b = torch.randn(1, 4, 1, 1, requires_grad=True)
    y = x + b
    return [x, b], y
