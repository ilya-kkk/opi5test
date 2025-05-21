FROM arm64v8/python:3.11-slim

WORKDIR /app

# 1) Системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
      curl \
      libopencv-dev \
      python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Скачиваем конкретную библиотеку рантайма RKNN для RK3588
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    git clone --depth 1 https://github.com/rockchip-linux/rknn-toolkit2.git /tmp/rknn-toolkit2 && \
    mkdir -p /usr/lib64 && \
    find /tmp/rknn-toolkit2 -name "librknnrt.so" -exec cp {} /usr/lib64/ \; && \
    find /tmp/rknn-toolkit2 -name "liblog.so" -exec cp {} /usr/lib64/ \; && \
    find /tmp/rknn-toolkit2 -name "lib*.so" -exec cp {} /usr/lib64/ \; && \
    chmod 755 /usr/lib64/*.so && \
    ldconfig && \
    rm -rf /tmp/rknn-toolkit2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3) Устанавливаем RKNN-Lite и основной RKNN-Toolkit2 + остальные Python‑зависимости
COPY rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl /tmp/
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --verbose /tmp/rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl && \
    pip install --no-cache-dir --verbose rknn-toolkit2 && \
    pip install --no-cache-dir --verbose -r /app/requirements.txt && \
    pip install --no-cache-dir --verbose pandas tabulate && \
    rm -rf /root/.cache/pip/*

# 4) Копируем код приложения
COPY . /app

# 5) Обеспечиваем, что динамический загрузчик найдёт librknnrt.so
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

# 6) Запуск инференс-скрипта
CMD ["python", "infer_test.py"]
