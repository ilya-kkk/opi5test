FROM python:3.9-slim

# Обновление и зависимости системы
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Установка pip-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
WORKDIR /app
COPY . .

# Установка прав доступа (если нужно использовать устройства)
RUN chmod -R 777 /app

# Точка входа по умолчанию
CMD ["python", "main.py"]
