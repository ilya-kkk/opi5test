FROM arm64v8/python:3.11-slim

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y cmake
RUN apt-get install -y build-essential
RUN apt-get install -y libssl-dev
RUN apt-get install -y libffi-dev
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN apt-get install -y libxrender-dev
RUN apt-get install -y wget
RUN apt-get install -y unzip
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt /tmp/

# Install Python packages and clean up in one layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt && \
    pip install --no-cache-dir rknn-toolkit2 && \
    rm -rf /root/.cache/pip/* 

# Copy application code
COPY . /app


