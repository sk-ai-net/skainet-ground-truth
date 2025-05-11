# Use Python 3.12.1 as base image
FROM python:3.12.1-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile
COPY Pipfile ./

# Install dependencies
RUN pipenv install

# Copy the rest of the application
COPY . .

# Set the default command
CMD ["pipenv", "run", "gt", "src", "results"]
