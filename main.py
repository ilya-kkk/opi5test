import os
import time
import cv2
import numpy as np
import torch
from rknn.api import RKNN
from roboflow import Roboflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# === Настройки ===
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION"))

PT_MODEL = 'v10nfull.pt'
ONNX_MODEL = 'model.onnx'
INPUT_SIZE = (640, 480)
TARGET_PLATFORM = 'rk3588'
QUANT_TYPES = ['fp16', 'int8', 'int4']

DATASET_DIR = 'calib_images'
TRAIN_TXT = 'dataset_train.txt'
VALID_TXT = 'dataset_valid.txt'

# === 0. Roboflow загрузка и сплит ===
def download_and_split_dataset():
    print('[INFO] Загружаем датасет с Roboflow...')
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("folder")

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

# === 1. .pt → ONNX ===
def convert_pt_to_onnx(pt_model, onnx_output):
    model = torch.load(pt_model)
    model.eval()
    dummy_input = torch.randn(1, 3, *INPUT_SIZE)
    torch.onnx.export(model, dummy_input, onnx_output, input_names=['input'], output_names=['output'], opset_version=11)
    print(f'[INFO] ONNX сохранён: {onnx_output}')

# === 2. ONNX → RKNN ===
def convert_onnx_to_rknn(onnx_path, out_path, quant_type):
    rknn = RKNN()
    rknn.config(target_platform=TARGET_PLATFORM,
                quantize_input=(quant_type != 'fp16'),
                quantized_dtype=quant_type if quant_type != 'fp16' else None)
    rknn.load_onnx(model=onnx_path)
    rknn.build(do_quantization=(quant_type != 'fp16'),
               dataset=TRAIN_TXT)
    rknn.export_rknn(out_path)
    rknn.release()
    print(f'[INFO] RKNN экспорт: {quant_type} → {out_path}')

# === 3. Инференс + время + точность ===
def evaluate_model(rknn_path):
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(target='rk3588')

    with open(VALID_TXT) as f:
        val_paths = [line.strip() for line in f.readlines()]

    total_time = 0
    correct = 0
    total = 0

    for path in val_paths[:50]:  # ограничим до 50 примеров для скорости
        img = cv2.imread(path)
        img = cv2.resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        start = time.time()
        outputs = rknn.inference(inputs=[img])[0]
        elapsed = time.time() - start
        total_time += elapsed

        pred = np.argmax(outputs)
        label = extract_label_from_filename(path)  # 👇 ниже определим

        if pred == label:
            correct += 1
        total += 1

    rknn.release()
    acc = correct / total * 100
    avg_time = (total_time / total) * 1000
    return avg_time, acc

# === 4. Поддержка label из имени файла ===
def extract_label_from_filename(path):
    # Пример: path = '.../dog_3.jpg' → label = 3
    filename = os.path.basename(path)
    for part in filename.split('_'):
        if part.split('.')[0].isdigit():
            return int(part.split('.')[0])
    return 0  # fallback

# === Главный код ===
if __name__ == '__main__':
    download_and_split_dataset()
    convert_pt_to_onnx(PT_MODEL, ONNX_MODEL)

    base_model_path = 'model_base.rknn'
    convert_onnx_to_rknn(ONNX_MODEL, base_model_path, 'fp16')

    quant_model_paths = []
    for quant in QUANT_TYPES:
        path = f'model_{quant}.rknn'
        convert_onnx_to_rknn(ONNX_MODEL, path, quant)
        quant_model_paths.append(path)

    print('\n[RESULTS]')
    models_to_test = [('fp16 (base)', base_model_path)] + list(zip(QUANT_TYPES, quant_model_paths))
    for label, path in models_to_test:
        time_ms, acc = evaluate_model(path)
        print(f'{label}: {time_ms:.2f} ms | Accuracy: {acc:.2f}%')
