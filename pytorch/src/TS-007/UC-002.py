import torch
from gt.core import Executable

# UC-002: Range slices with steps (positive/negative, open-ended)

@Executable("Rows ::2 (every 2nd), all columns")
def range_every_second_row():
    x = torch.arange(1, 25, dtype=torch.float32, requires_grad=True).reshape(6, 4)
    # shape: (6, 4) -> (3, 4)
    y = x[::2, :]
    return [x], y

@Executable("All rows, columns ::-1 (reverse columns)")
def range_reverse_columns():
    x = torch.arange(1, 17, dtype=torch.float32, requires_grad=True).reshape(4, 4)
    # shape: (4, 4) -> (4, 4)
    y = x[:, ::-1]
    return [x], y

@Executable("Rows 1:5:2 and columns 0:4:2")
def range_strided_submatrix():
    x = torch.arange(0, 64, dtype=torch.float32, requires_grad=True).reshape(8, 8)
    # shape: (8, 8) -> (2, 2)
    y = x[1:5:2, 0:4:2]
    return [x], y

@Executable("Reverse rows (::-1), keep columns :")
def range_reverse_rows():
    x = torch.arange(0, 12, dtype=torch.float32, requires_grad=True).reshape(3, 4)
    # shape: (3, 4) -> (3, 4)
    y = x[::-1, :]
    return [x], y
