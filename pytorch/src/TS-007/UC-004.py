import torch
from gt.core import Executable

# UC-004: First/last/second-last selections, ellipsis usage

@Executable("First row, all columns")
def first_row_all_columns():
    x = torch.arange(0, 20, dtype=torch.float32, requires_grad=True).reshape(5, 4)
    # shape: (5, 4) -> (4,)
    y = x[0, :]
    return [x], y

@Executable("Last row (index -1), all columns")
def last_row_all_columns():
    x = torch.arange(0, 20, dtype=torch.float32, requires_grad=True).reshape(5, 4)
    # shape: (5, 4) -> (4,)
    y = x[-1, :]
    return [x], y

@Executable("All rows, second-last column (index -2)")
def second_last_column():
    x = torch.arange(0, 20, dtype=torch.float32, requires_grad=True).reshape(5, 4)
    # shape: (5, 4) -> (5,)
    y = x[:, -2]
    return [x], y

@Executable("Use ellipsis to get last channel: x[..., -1]")
def ellipsis_last_channel():
    # Shape: (B,C,H,W) = (2,3,4,4)
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    # shape: (2, 3, 4, 4) -> (2, 3, 4)
    y = x[..., -1]
    return [x], y

@Executable("Use ellipsis to select center spatial row across all dims: x[..., 2, :]")
def ellipsis_center_spatial_row():
    # Shape: (B,C,H,W) = (2,3,5,6)
    x = torch.randn(2, 3, 5, 6, requires_grad=True)
    # shape: (2, 3, 5, 6) -> (2, 3, 6)
    y = x[..., 2, :]
    return [x], y
