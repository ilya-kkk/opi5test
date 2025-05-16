FROM arm64v8/python:3.8-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов
COPY requirements.txt /tmp/
COPY rknn_toolkit2-1.4.0-cp38-cp38-linux_aarch64.whl /tmp/

# Установка pip-зависимостей
RUN pip install --upgrade pip
RUN pip install --no-cache-dir /tmp/rknn_toolkit2-1.4.0-cp38-cp38-linux_aarch64.whl
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Копируем всё остальное
COPY . /app

