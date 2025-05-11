# SKAINet Ground Truth - PyTorch

This repository contains PyTorch-based ground truth generation for SKAINet.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for Python package management and virtual environments.

## Docker Setup

### Building the Docker Image

To build the Docker image, run the following command from the `pytorch` directory:

```bash
docker build -t skainet-ground-truth .
```

### Running the Container

To run the container and generate ground truth data, use:

```bash
docker run -v "$(pwd)/src:/app/src" -v "$(pwd)/results:/app/results" skainet-ground-truth
```

This command will:
- Mount your local `src` directory into the container for input
- Mount your local `results` directory for output
- Run the ground truth generation process
- Save the results in GGUF format in the `results/TS-001` directory

The generated files include ground truth data for various convolution operations:
- Basic 2D convolution
- Strided convolution
- Padded convolution
- Depthwise convolution
- Batched input
