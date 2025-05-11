import torch
from  gt.core import Executable

#@Executable("Small Input Size")
#def use_case_5():
#    x = torch.randn(1, 3, 2, 2, requires_grad=True)
#    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
#    y = conv(x)  # May raise an error due to small input size
#    return [x], y
