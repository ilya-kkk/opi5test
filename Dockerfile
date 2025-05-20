FROM arm64v8/python:3.11-slim

WORKDIR /app

# Increase apt space
RUN echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/90assumeyes && \
    echo 'APT::Get::AllowUnauthenticated "true";' > /etc/apt/apt.conf.d/90allowunauth && \
    echo 'APT::Get::Fix-Missing "true";' > /etc/apt/apt.conf.d/90fixmissing

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    wget \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*

# Install OpenCV dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*

# Install development libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libssl-dev \
    libffi-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libhdf5-dev \
    libatlas-base-dev \
    libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*

# Install RKNN Runtime
RUN pip install --no-cache-dir rknn-toolkit2 && \
    pip install --no-cache-dir rknn-toolkit-lite2

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python packages and clean up in one layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip/*

# Copy application code
COPY . /app


