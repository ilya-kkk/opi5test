FROM arm64v8/ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    unzip \
    && apt-get clean

# Обновляем pip и ставим зависимости
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

COPY requirements.txt /tmp/
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Копируем проект
WORKDIR /app
COPY . /app

# Установка переменных среды
ENV RKNN_TOOLKIT2_ENABLE_NPU_QUANT=true


