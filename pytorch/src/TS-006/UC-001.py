import torch
from gt.core import Executable

"""
UC-001: Broadcasting with scalar (addition)

Covers:
- Operation: addition (+)
- Layout: NCHW
- Batch usage: yes (B > 1)
- Broadcasting pattern: scalar -> (B,C,H,W)
"""


@Executable("Addition with scalar broadcasting (NCHW, batched)")
def add_scalar_broadcast_nchw_batched():
    # Input tensor in NCHW with batch
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    # Scalar that will broadcast across all dimensions
    s = torch.tensor(0.5, requires_grad=True)
    y = x + s
    return [x, s], y
