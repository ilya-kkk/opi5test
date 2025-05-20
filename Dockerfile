FROM arm64v8/python:3.11-slim

WORKDIR /app

# 1) Системные зависимости + NPU runtime
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
      python3-opencv \
      # вот здесь ставим runtime для RK3588 NPU:
      rknpu2-rk3588 \
      python3-rknnlite2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# 2) Копируем и устанавливаем RKNN Lite wheel
COPY rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir /tmp/rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl && \
    rm -rf /root/.cache/pip/*

# 3) Устанавливаем основной rknn-toolkit2 и остальные зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir rknn-toolkit2 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir pandas tabulate && \
    rm -rf /root/.cache/pip/*

# 4) Копируем ваш код
COPY . /app

# 5) Добавляем LD_LIBRARY_PATH на случай, если librknnrt.so ставится в нестандартную папку
ENV LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH

# 6) Точка входа
CMD ["python", "infer_test.py"]
