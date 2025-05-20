import os
import time
import cv2
import numpy as np
from rknn.api import RKNN
import argparse
from tqdm import tqdm

def load_model(model_path):
    """Загрузка RKNN модели"""
    rknn = RKNN()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f'[ERROR] Load RKNN model failed: {ret}')
        return None

    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print(f'[ERROR] Init runtime environment failed: {ret}')
        return None

    return rknn

def preprocess_image(image_path, input_size=(640, 640)):
    """Предобработка изображения"""
    img = cv2.imread(image_path)
    if img is None:
        print(f'[ERROR] Could not read image: {image_path}')
        return None

    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)

def run_inference(model, image, num_runs=100):
    """Запуск инференса и замер времени"""
    times = []
    
    # Прогрев модели
    for _ in range(10):
        model.inference(inputs=[image])
    
    # Замер времени
    for _ in tqdm(range(num_runs), desc="Running inference"):
        start = time.time()
        outputs = model.inference(inputs=[image])
        end = time.time()
        times.append((end - start) * 1000)  # в миллисекундах
    
    return times

def main():
    parser = argparse.ArgumentParser(description='RKNN Model Inference Test')
    parser.add_argument('--model', type=str, required=True, help='Path to RKNN model')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--runs', type=int, default=100, help='Number of inference runs')
    args = parser.parse_args()

    # Загрузка модели
    print(f'[INFO] Loading model: {args.model}')
    model = load_model(args.model)
    if model is None:
        return

    # Загрузка и предобработка изображения
    print(f'[INFO] Loading image: {args.image}')
    image = preprocess_image(args.image)
    if image is None:
        return

    # Запуск инференса
    print(f'[INFO] Running inference {args.runs} times...')
    times = run_inference(model, image, args.runs)

    # Вывод результатов
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print('\n[RESULTS]')
    print(f'Average inference time: {avg_time:.2f} ms')
    print(f'Std deviation: {std_time:.2f} ms')
    print(f'Min time: {min_time:.2f} ms')
    print(f'Max time: {max_time:.2f} ms')
    print(f'FPS: {1000/avg_time:.2f}')

    # Очистка
    model.release()

if __name__ == '__main__':
    main()
