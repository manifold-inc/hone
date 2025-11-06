FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy inference script
COPY inference.py .

# Make it executable
RUN chmod +x inference.py

# Set environment
ENV PYTHONUNBUFFERED=1

CMD ["python3", "inference.py"]