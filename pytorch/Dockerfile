# Use Python 3.12.1 as base image
FROM python:3.12.1-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml requirements.txt ./

# Create venv and install dependencies using uv
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set the default command
CMD [".venv/bin/gt", "src", "results"]
