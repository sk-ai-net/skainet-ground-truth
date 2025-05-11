import torch
from  gt.core import Executable

@Executable("Basic 2D Convolution")
def basic_2D_convolution():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)  # Input: (batch_size, channels, height, width)
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
    y = conv(x)
    return [x], y


@Executable("Strided Convolution")
def strided_convolution():
    x = torch.randn(1, 3, 32, 32, requires_grad=True) # Input: (batch_size, channels, height, width)
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0)
    y = conv(x)    
    return [x], y