import torch
from gt.core import Executable

# UC-001: Simple basics (rows/cols, single-dim slices)

@Executable("Take all rows, columns 1: (open-ended)")
def simple_cols_from_1():
    x = torch.arange(1, 10, dtype=torch.float32, requires_grad=True).reshape(3, 3)
    # shape: (3, 3) -> (3, 2)
    y = x[:, 1:]
    return [x], y

@Executable("Take rows 1:, columns 1:")
def simple_rows_and_cols_from_1():
    x = torch.arange(1, 10, dtype=torch.float32, requires_grad=True).reshape(3, 3)
    # shape: (3, 3) -> (2, 2)
    y = x[1:, 1:]
    return [x], y

@Executable("Take middle row (index 1) all columns")
def simple_middle_row():
    x = torch.arange(1, 10, dtype=torch.float32, requires_grad=True).reshape(3, 3)
    # shape: (3, 3) -> (3,)
    y = x[1, :]
    return [x], y

@Executable("Take all rows, middle column (index 1)")
def simple_middle_col():
    x = torch.arange(1, 10, dtype=torch.float32, requires_grad=True).reshape(3, 3)
    # shape: (3, 3) -> (3,)
    y = x[:, 1]
    return [x], y
