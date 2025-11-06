# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create model cache directory with proper permissions
# This directory will persist between prep and inference phases
RUN mkdir -p /app/models && \
    chmod 777 /app/models

# Set environment variables for Hugging Face caching
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/models
ENV PYTHONUNBUFFERED=1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy inference script
COPY inference.py .

# Make inference script executable
RUN chmod +x inference.py

# Default command (will be overridden by sandbox runner)
CMD ["python3", "inference.py"]