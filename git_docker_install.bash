#!/bin/bash

set -e

echo "🔄 Обновление системы..."
sudo apt update && sudo apt upgrade -y

echo "🔧 Установка Git..."
sudo apt install -y git

echo "🐳 Установка Docker..."

# Установка зависимостей
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Добавление ключа Docker GPG
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Добавление Docker репозитория
echo \
  "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Установка Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "👤 Добавление пользователя в группу docker..."
sudo usermod -aG docker $USER

echo "✅ Проверка Docker версии:"
docker --version

echo "📦 Установка Docker Compose CLI (если нужно отдельно)..."
sudo apt install -y docker-compose

echo "🎉 Готово! Перезагрузите систему или выполните:"
echo "   exec su -l $USER"
