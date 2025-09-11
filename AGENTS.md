# SKaiNET Ground Truth Development Guidelines


## Project Overview
This project generates structured ground truth data for validating calculations used in machine learning framework by using PyTorch for executing  and the gradienttracer framework. The data is stored in GGUF format for use in neural network testing and validation.

## Build/Configuration Instructions

### Prerequisites
- Python 3.12.1 or higher
- [uv](https://github.com/astral-sh/uv) for Python package management

### Local Development Setup

1. **Navigate to the pytorch directory:**
   ```bash
   cd pytorch
   ```

2. **Install uv (if not already installed):**
   ```bash
   pip install uv
   ```

3. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

4. **Additional dependency for full functionality:**
   ```bash
   uv pip install torchvision
   ```
   *Note: torchvision is required for some test suites but not listed in requirements.txt*

### Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build -t skainet-ground-truth .
   ```

2. **Run the container:**
   ```bash
   docker run -v "$(pwd)/src:/app/src" -v "$(pwd)/results:/app/results" skainet-ground-truth
   ```

### Key Dependencies
- **gradienttracer**: Custom framework from `git+https://github.com/sk-ai-net/gradienttracer.git`
- **gguf**: For GGUF file format handling
- **torch**: PyTorch framework (automatically installs appropriate version)
- **torchvision**: Required for some test suites (not in requirements.txt)

## Testing Information

### Test Structure
The project uses a hierarchical test organization:
- `src/TS-XXX/UC-YYY.py`: Test suites (TS) containing use cases (UC)
- Each test file can contain multiple test functions decorated with `@Executable`

### Current Test Suites
- **TS-001**: Convolution operations (basic 2D, strided, padded, depthwise, batched)
- **TS-002**: Tensor slicing operations
- **TS-003**: Tensor flatten operations
- **TS-004**: GGUF string operations
- **TS-005**: MNIST training and model serialization

### Running Tests

1. **Execute all test suites:**
   ```bash
   source .venv/bin/activate
   gt src results
   ```

2. **Execute specific test suite:**
   ```bash
   gt src/TS-001 results
   ```

### Test Output
- Results are stored in GGUF format in the specified output directory
- File naming: `TS_XXX_UC_YYY.function_name.gguf`
- Directory structure mirrors the source structure

### Creating New Tests

1. **Test file structure:**
   ```python
   import torch
   from gt.core import Executable

   @Executable("Descriptive Test Name")
   def test_function():
       # Create input tensors with requires_grad=True for gradient tracking
       x = torch.randn(2, 3, requires_grad=True)
       
       # Perform operations
       result = torch.nn.functional.relu(x)
       
       # Return: ([input_tensors], output_tensor)
       return [x], result
   ```

2. **Test function requirements:**
    - Must be decorated with `@Executable("Description")`
    - Must return `([input_tensors], output_tensor)` tuple
    - Input tensors should have `requires_grad=True` for gradient computation
    - Function names should be descriptive and unique within the file

3. **Example test execution:**
   ```python
   # Create a simple test file: src/TS-DEMO/UC-001.py
   import torch
   from gt.core import Executable

   @Executable("Basic Matrix Multiplication")
   def basic_matrix_multiplication():
       x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
       y = torch.tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
       result = torch.mm(x, y)
       return [x, y], result
   ```

## Development Information

### Code Style Guidelines
- Follow standard Python naming conventions (snake_case for functions)
- Use descriptive function names that clearly indicate the operation being tested
- Include docstrings for complex operations
- Maintain consistent tensor shapes and data types within test suites

### Project Structure
```
pytorch/
├── src/                    # Test source files
│   ├── TS-001/            # Convolution tests
│   ├── TS-002/            # Slicing tests
│   ├── TS-003/            # Flatten tests
│   ├── TS-004/            # GGUF tests
│   └── TS-005/            # MNIST tests
├── results/               # Generated ground truth data (GGUF files)
├── data/                  # Training data (e.g., MNIST)
├── pyproject.toml         # Project configuration
├── requirements.txt       # Python dependencies
└── Dockerfile            # Docker configuration
```

### Debugging Tips
- Use `print()` statements in test functions for debugging (output visible during gt execution)
- Tensor shapes are automatically printed for debugging in some test suites
- Check GGUF file generation in results directory to verify test execution
- Missing dependencies (like torchvision) will cause runtime errors during gt execution

### Common Pitfalls
- **Missing torchvision**: Install separately as it's not in requirements.txt but required by some tests
- **Incorrect return format**: Must return `([inputs], output)` tuple format
- **Missing requires_grad**: Input tensors need `requires_grad=True` for gradient computation
- **Import errors**: Ensure all required modules are imported (torch, gt.core.Executable)

### Integration with gradienttracer
- The `gt` command processes all `@Executable` decorated functions
- Input tensors and outputs are serialized to GGUF format for validation
- The gradienttracer framework handles automatic gradient computation and result storage
- Results can be used for validating SKAINet library implementations