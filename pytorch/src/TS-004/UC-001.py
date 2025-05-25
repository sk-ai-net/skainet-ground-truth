import gguf
import torch
from  gt.core import Executable

@Executable("simple small gguf file with string value")
def simple_gguf_string():
    # Create a new GGUF writer instance
    writer = gguf.GGUFWriter("skainet_small.gguf", arch="generic")

    # Add a string property named "model_name"
    writer.add_string("model_name", "skainet-small")

    # Finalize and write the GGUF file
    writer.write_header_to_file()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)  # random tensor to keep up with API
    return [x], x
