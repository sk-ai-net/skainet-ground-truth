import torch
import torch.nn as nn
from  gt.core import Executable

@Executable("MNIST flatten")
def image_flatten():
    flatten = nn.Flatten()
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    y = flatten(x)
    print(f"Shape {y.shape}") #(2, 784)
    return [x], y

@Executable("Flatten with with custom start dim")
def flatten_with_custom_start_dim():
    flatten = nn.Flatten(start_dim=1)
    input_tensor = torch.randn(2, 3, 4)
    output_tensor = flatten(input_tensor)
    print(output_tensor.shape)  # 3*4 = 12
    return [input_tensor], output_tensor



@Executable("Flatten single sample")
def flatten_single_sample(self):
    flatten = nn.Flatten()
    input_tensor = torch.randn(1, 3, 3)
    output_tensor = flatten(input_tensor)
    print(output_tensor.shape)  # (1, 9)
    return [input_tensor], output_tensor


@Executable("Flatten preserve batch dim")
def flatten_preserve_batch_dim(self):
    flatten = nn.Flatten()
    input_tensor = torch.randn(10, 5, 2, 2)
    output_tensor = flatten(input_tensor)
    # self.assertEqual(output_tensor.shape, (10, 20))  # 5*2*2 = 20
    return [input_tensor], output_tensor

@Executable("Flatten no batch dim")
def flatten_no_batch_dim(self):
    flatten = nn.Flatten()
    input_tensor = torch.randn(3, 4)
    output_tensor = flatten(input_tensor)
    # self.assertEqual(output_tensor.shape, (1, 12))  # default start_dim=1, input treated as (1, 3, 4)
    return [input_tensor], output_tensor


  