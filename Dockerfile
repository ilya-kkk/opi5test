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
      libopencv-dev \
      python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Скачиваем конкретную библиотеку рантайма RKNN для RK3588
RUN curl -L --retry 3 --retry-delay 2 --retry-max-time 30 -o /usr/lib/librknnrt.so \
      https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/master/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig

# 3) Устанавливаем RKNN-Lite и основной RKNN-Toolkit2 + остальные Python‑зависимости
COPY rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl /tmp/
COPY requirements.txt /app/
RUN pip install --no-cache-dir /tmp/rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir rknn-toolkit2 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir pandas tabulate && \
    rm -rf /root/.cache/pip/*

# 4) Копируем код приложения
COPY . /app

# 5) Обеспечиваем, что динамический загрузчик найдёт librknnrt.so
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# 6) Запуск инференс-скрипта
CMD ["python", "infer_test.py"]
