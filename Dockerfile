FROM arm64v8/python:3.11-slim

WORKDIR /app

# 1) Устанавливаем системные зависимости
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
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Склонировать RKNN-Toolkit2 и скопировать librknnrt.so для RK3588
RUN git clone --depth 1 https://github.com/rockchip-linux/rknn-toolkit2.git /tmp/rknn-toolkit2 && \
    cp /tmp/rknn-toolkit2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/ && \
    ldconfig && \
    rm -rf /tmp/rknn-toolkit2

# 3) Устанавливаем RKNN‑Lite wheel (если нужен) и rknn-toolkit2 из PyPI
COPY rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir /tmp/rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl && \
    rm -rf /root/.cache/pip/*

# 4) Устанавливаем rknn-toolkit2 и остальные Python-зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir rknn-toolkit2 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir pandas tabulate && \
    rm -rf /root/.cache/pip/*

# 5) Копируем код приложения
COPY . /app

# 6) Обеспечиваем поиск библиотек в /usr/lib
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# 7) Запускаем тестовый скрипт
CMD ["python", "infer_test.py"]
