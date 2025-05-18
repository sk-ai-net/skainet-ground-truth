import torch
from  gt.core import Executable

@Executable("take all rows, take columns starting from index 1 to the end")
def slice_1():
    # Create a 3x3 tensor with values from 1 to 9
    x = torch.arange(1, 10).reshape(3, 3)
    print(x)
    y = x[:,1:]
    print(y)
    return [x], y

@Executable("take rows starting with the index 1, take columns starting from index 1 to the end")
def slice_2():
    # Create a 3x3 tensor with values from 1 to 9
    x = torch.arange(1, 10).reshape(3, 3)
    print(x)
    y = x[1:,1:]
    print(y)
    return [x], y


@Executable("Take the rows from the second to the second last and all columns until the second last one")
def slice_3():
    # Create a 3x3 tensor with values from 1 to 9
    x = torch.arange(1, 10).reshape(3, 3)
    print(x)
    y = x[1:-1,:-2]
    print(y)
    return [x], y