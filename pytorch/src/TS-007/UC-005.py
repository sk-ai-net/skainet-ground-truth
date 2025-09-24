import torch
from gt.core import Executable

# UC-005: Typical ML slices on NCHW tensors (channel subset, crops, sequence window)

@Executable("NCHW: select channel subset 1:3 across all batches and spatial")
def ml_channel_subset():
    # Shape: (B,C,H,W)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    # shape: (2, 4, 8, 8) -> (2, 2, 8, 8)
    y = x[:, 1:3, :, :]
    return [x], y

@Executable("NCHW: center crop HxW from 32x32 to 16x16")
def ml_center_crop():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    h0, w0 = 8, 8  # start index for 16x16 crop
    h1, w1 = h0 + 16, w0 + 16
    # shape: (1, 3, 32, 32) -> (1, 3, 16, 16)
    y = x[:, :, h0:h1, w0:w1]
    return [x], y

@Executable("NCHW: remove 1-pixel border -> H-2 x W-2")
def ml_border_crop():
    x = torch.randn(1, 3, 10, 12, requires_grad=True)
    # shape: (1, 3, 10, 12) -> (1, 3, 8, 10)
    y = x[:, :, 1:-1, 1:-1]
    return [x], y

@Executable("Sequence window: select timesteps 2:6 from (B,T,C)")
def ml_sequence_window():
    x = torch.randn(2, 10, 16, requires_grad=True)  # (batch, time, features)
    # shape: (2, 10, 16) -> (2, 4, 16)
    y = x[:, 2:6, :]
    return [x], y
