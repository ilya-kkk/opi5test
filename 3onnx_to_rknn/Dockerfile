FROM arm64v8/python:3.11-slim

WORKDIR /app

# 1) Системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake build-essential libssl-dev libffi-dev \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
      wget curl libopencv-dev python3-opencv git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Скопировать runtime-библиотеки из rknpu2
RUN git clone --depth 1 https://github.com/rockchip-linux/rknpu2.git /tmp/rknpu2 && \
    mkdir -p /usr/lib64 && \
    find /tmp/rknpu2/runtime/RK3588/Linux/librknn_api/aarch64 \
         -name "*.so" -exec cp {} /usr/lib64/ \; && \
    chmod 755 /usr/lib64/*.so && ldconfig && \
    rm -rf /tmp/rknpu2

# 3) Устанавливаем RKNN-Lite и основной RKNN-Toolkit2 + остальные Python‑зависимости
COPY rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl /tmp/
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/rknn_toolkit_lite2-1.6.0-cp311-cp311-linux_aarch64.whl && \
    pip install --no-cache-dir rknn-toolkit2 && \
    pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 && \
    pip install --no-cache-dir ultralytics==8.0.196 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir pandas tabulate && \
    pip install --no-cache-dir onnx onnx-graphsurgeon && \
    rm -rf /root/.cache/pip/*

# 4) Копируем код
COPY . /app

# 5) Пути к библиотекам
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

# 6) Запуск
CMD ["python", "convert.py"]
