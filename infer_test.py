import os
import time
import cv2
import numpy as np
from rknn.api import RKNN
import argparse
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

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
    print("Warming up model...")
    for _ in range(10):
        model.inference(inputs=[image])
    
    # Замер времени
    print("Running inference tests...")
    for _ in tqdm(range(num_runs), desc="Running inference"):
        start = time.time()
        outputs = model.inference(inputs=[image])
        end = time.time()
        times.append((end - start) * 1000)  # в миллисекундах
    
    return times

def test_model(model_path, image, num_runs=100):
    """Тестирование одной модели"""
    print(f'\n[INFO] Testing model: {os.path.basename(model_path)}')
    
    # Загрузка модели
    model = load_model(model_path)
    if model is None:
        return None

    # Запуск инференса
    times = run_inference(model, image, num_runs)
    
    # Расчет статистики
    stats = {
        'Model': os.path.basename(model_path),
        'Avg Time (ms)': np.mean(times),
        'Std Dev (ms)': np.std(times),
        'Min Time (ms)': np.min(times),
        'Max Time (ms)': np.max(times),
        'FPS': 1000/np.mean(times)
    }
    
    # Очистка
    model.release()
    return stats

def main():
    parser = argparse.ArgumentParser(description='RKNN Model Inference Test')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--runs', type=int, default=100, help='Number of inference runs')
    args = parser.parse_args()

    # Список моделей для тестирования
    models = [
        'model_fp16.rknn',
        'model_int8.rknn',
        'model_int4.rknn'
    ]

    # Загрузка и предобработка изображения
    print(f'[INFO] Loading image: {args.image}')
    image = preprocess_image(args.image)
    if image is None:
        return

    # Тестирование всех моделей
    results = []
    for model_path in models:
        if not os.path.exists(model_path):
            print(f'[WARNING] Model {model_path} not found, skipping...')
            continue
            
        stats = test_model(model_path, image, args.runs)
        if stats:
            results.append(stats)

    if not results:
        print('[ERROR] No models were successfully tested')
        return

    # Вывод результатов в виде таблицы
    print('\n[RESULTS]')
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt='.2f'))

    # Сохранение результатов в CSV
    csv_path = 'inference_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n[INFO] Results saved to {csv_path}')

if __name__ == '__main__':
    main()
