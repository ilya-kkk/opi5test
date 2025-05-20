FROM arm64v8/python:3.11-slim

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    unzip \
    libopencv-dev \
    python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*

# Install RKNN Runtime
RUN wget --no-check-certificate https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v1.5.0/rknn-toolkit2-1.5.0-cp311-cp311-linux_aarch64.whl -O rknn-toolkit2.whl && \
    wget --no-check-certificate https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v1.5.0/rknn-toolkit-lite2-1.5.0-cp311-cp311-linux_aarch64.whl -O rknn-toolkit-lite2.whl && \
    pip install --no-cache-dir rknn-toolkit2.whl && \
    pip install --no-cache-dir rknn-toolkit-lite2.whl && \
    rm *.whl

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python packages and clean up in one layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip/*

# Copy application code
COPY . /app


