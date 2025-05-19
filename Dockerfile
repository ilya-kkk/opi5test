FROM arm64v8/python:3.11-slim

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt /tmp/

# Install Python packages and clean up in one layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir rknn-toolkit2 && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Copy application code
COPY . /app


