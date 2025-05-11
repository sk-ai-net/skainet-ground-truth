import torch
from  gt.core import Executable

# RuntimeError: Input type (double) and bias type (float) should be the same


#@Executable("Different Data Types (float64)")
#def use_case_7():
#    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.float64)
#    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
#    y = conv(x)
#    return [x], y