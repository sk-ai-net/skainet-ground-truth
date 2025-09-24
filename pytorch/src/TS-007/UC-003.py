import torch
from gt.core import Executable

# UC-003: Explicit start/end bounds and None

@Executable("Explicit start=0, end=-1 on rows; all columns")
def bounds_start0_end_minus1_rows():
    x = torch.arange(0, 20, dtype=torch.float32, requires_grad=True).reshape(5, 4)
    # shape: (5, 4) -> (4, 4)
    y = x[0:-1, :]
    return [x], y

@Executable("Rows 1:-1, columns :3 (open start)")
def bounds_open_start_col():
    x = torch.arange(0, 36, dtype=torch.float32, requires_grad=True).reshape(6, 6)
    # shape: (6, 6) -> (4, 3)
    y = x[1:-1, :3]
    return [x], y

@Executable("Use full open slice :, and explicit 0: on columns")
def bounds_full_and_zero_start():
    x = torch.arange(0, 16, dtype=torch.float32, requires_grad=True).reshape(4, 4)
    # shape: (4, 4) -> (4, 4)
    y = x[:, 0:]
    return [x], y

@Executable("Mix: rows 2:, cols :-2")
def bounds_mix():
    x = torch.arange(0, 30, dtype=torch.float32, requires_grad=True).reshape(5, 6)
    # shape: (5, 6) -> (3, 4)
    y = x[2:, :-2]
    return [x], y
