FROM arm64v8/python:3.11-slim  

WORKDIR /app

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
    unzip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt  

RUN pip install rknn-toolkit2

COPY . /app


