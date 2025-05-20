import os
import time
import cv2
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from rknn.api import RKNN
import ctypes
import sys
import glob


# === Настройки ===
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION"))

ONNX_MODEL = 'v10nint8256.onnx'  # Указываем заранее сконвертированную модель
INPUT_SIZE = (640, 640)
TARGET_PLATFORM = 'rk3588'
QUANT_TYPES = ['fp16', 'int8']  # Убрал int4, так как он может не поддерживаться

DATASET_DIR = 'calib_images'
TRAIN_TXT = 'dataset_train.txt'
VALID_TXT = 'dataset_valid.txt'


# === 0. Roboflow загрузка и сплит ===
def download_and_split_dataset():
    print('[INFO] Загружаем датасет с Roboflow...')
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov5")

    os.makedirs(DATASET_DIR, exist_ok=True)

    all_images = []
    for root, _, files in os.walk(dataset.location):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(root, file)
                dst = os.path.join(DATASET_DIR, file)
                if not os.path.exists(dst):
                    os.system(f'cp "{src}" "{dst}"')
                all_images.append(os.path.abspath(dst))

    train_set, valid_set = train_test_split(all_images, train_size=0.8, random_state=42)

    with open(TRAIN_TXT, 'w') as f:
        f.write('\n'.join(train_set))
    with open(VALID_TXT, 'w') as f:
        f.write('\n'.join(valid_set))

    print(f'[INFO] Сплит завершён. Train: {len(train_set)}, Valid: {len(valid_set)}')

# === 1. ONNX → RKNN ===
def convert_onnx_to_rknn(onnx_path, out_path, quant_type):
    if not check_rknn_library():
        return False
        
    try:
        rknn = RKNN()
        rknn.config(target_platform=TARGET_PLATFORM)

        print(f'[INFO] Загружаем модель: {onnx_path}')
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print(f'[ERROR] Load ONNX model failed: {ret}')
            return False

        print(f'[INFO] Строим модель с квантизацией: {quant_type}')
        ret = rknn.build(do_quantization=(quant_type != 'fp16'), dataset=TRAIN_TXT)
        if ret != 0:
            print(f'[ERROR] Build model failed: {ret}')
            return False

        print(f'[INFO] Сохраняем RKNN: {out_path}')
        ret = rknn.export_rknn(out_path)
        if ret != 0:
            print(f'[ERROR] Export RKNN model failed: {ret}')
            return False

        rknn.release()
        return True
    except Exception as e:
        print(f'[ERROR] Exception during model conversion: {str(e)}')
        return False

# === 2. Инференс + время + точность ===
def evaluate_model(rknn_path):
    if not check_rknn_library():
        return 0, 0

    try:
        rknn = RKNN()
        ret = rknn.load_rknn(rknn_path)
        if ret != 0:
            print(f'[ERROR] Load RKNN model failed: {ret}')
            return 0, 0

        # Initialize runtime with target platform
        ret = rknn.init_runtime(target=TARGET_PLATFORM)
        if ret != 0:
            print(f'[ERROR] Init runtime environment failed: {ret}')
            return 0, 0

        # Список валидационных изображений
        with open(VALID_TXT) as f:
            val_paths = [p.strip() for p in f if p.strip()]

        total_time = correct = total = 0

        for path in val_paths[:50]:
            img = cv2.imread(path)
            if img is None:
                print(f'[WARNING] Could not read image: {path}')
                continue

            img = cv2.resize(img, INPUT_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

            start = time.time()
            outputs = rknn.inference(inputs=[img])
            if outputs is None:
                print(f'[WARNING] Inference failed for image: {path}')
                continue

            outputs = outputs[0]
            total_time += time.time() - start

            pred = int(np.argmax(outputs))
            label = extract_label_from_filename(path)
            if pred == label:
                correct += 1
            total += 1

        rknn.release()

        if total == 0:
            return 0, 0

        acc = correct / total * 100
        avg_time = (total_time / total) * 1000  # в мс
        return avg_time, acc
    except Exception as e:
        print(f'[ERROR] Exception during model evaluation: {str(e)}')
        return 0, 0

# === 3. Получение label из имени файла ===
def extract_label_from_filename(path):
    filename = os.path.basename(path)
    for part in filename.split('_'):
        if part.split('.')[0].isdigit():
            return int(part.split('.')[0])
    return 0

# === Главный код ===
if __name__ == '__main__':
    if not check_rknn_library():
        sys.exit(1)

    download_and_split_dataset()

    quant_model_paths = []
    for quant in QUANT_TYPES:
        path = f'model_{quant}.rknn'
        if convert_onnx_to_rknn(ONNX_MODEL, path, quant):
            quant_model_paths.append(path)

    if not quant_model_paths:
        print('[ERROR] No models were successfully converted')
        sys.exit(1)

    print('\n[RESULTS]')
    for label, path in zip(QUANT_TYPES, quant_model_paths):
        time_ms, acc = evaluate_model(path)
        print(f'{label}: {time_ms:.2f} ms | Accuracy: {acc:.2f}%')
