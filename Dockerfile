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

# Copy RKNN Toolkit wheel file
COPY rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir /tmp/rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl && \
    rm -rf /root/.cache/pip/*


COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir rknn-toolkit2 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip/*

RUN pip install pandas tabulate

# RUN apt update && apt install rknpu2-rk3588

ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

