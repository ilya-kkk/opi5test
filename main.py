import torch
import torchvision.models as models
import onnx
import numpy as np
import cv2
import time
from rknn.api import RKNN

# === Настройки ===
PT_MODEL = 'v10nfull.pt'
ONNX_MODEL = 'model.onnx'
IMAGE_PATH = 'test.jpg'
INPUT_SIZE = (224, 224)  # адаптируй под свою модель
TARGET_PLATFORM = 'rk3588'
QUANT_TYPES = ['fp16', 'int8', 'int4']

# === 1. Конвертация .pt в ONNX ===
def convert_pt_to_onnx(pt_model, onnx_output):
    model = torch.load(pt_model)
    model.eval()
    dummy_input = torch.randn(1, 3, *INPUT_SIZE)
    torch.onnx.export(model, dummy_input, onnx_output, input_names=['input'], output_names=['output'], opset_version=11)
    print(f'[INFO] ONNX model saved to {onnx_output}')

# === 2. Конвертация ONNX → RKNN с квантованием ===
def convert_onnx_to_rknn(onnx_path, out_path, quant_type):
    rknn = RKNN()
    rknn.config(target_platform=TARGET_PLATFORM,
                quantize_input=(quant_type != 'fp16'),
                quantized_dtype=quant_type if quant_type != 'fp16' else None)
    rknn.load_onnx(model=onnx_path)
    rknn.build(do_quantization=(quant_type != 'fp16'),
               dataset='./dataset.txt')  # Создай текстовый файл с путями к изображениям для калибровки INT8/INT4
    rknn.export_rknn(out_path)
    rknn.release()
    print(f'[INFO] Exported {quant_type} model to {out_path}')

# === 3. Инференс и замер времени ===
def run_inference(rknn_path):
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    rknn.init_runtime()
    
    # Подготовка изображения
    img = cv2.imread(IMAGE_PATH)
    img = cv2.resize(img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)

    # Прогрев
    for _ in range(3):
        rknn.inference(inputs=[img])

    # Замер
    start = time.time()
    rknn.inference(inputs=[img])
    elapsed = time.time() - start

    rknn.release()
    return elapsed * 1000  # в миллисекундах

# === Главный код ===
if __name__ == '__main__':
    convert_pt_to_onnx(PT_MODEL, ONNX_MODEL)

    base_model_path = 'model_base.rknn'
    convert_onnx_to_rknn(ONNX_MODEL, base_model_path, 'fp16')  # Базовая модель

    quant_model_paths = []
    for quant in QUANT_TYPES:
        path = f'model_{quant}.rknn'
        convert_onnx_to_rknn(ONNX_MODEL, path, quant)
        quant_model_paths.append(path)

    # Инференс всех моделей
    print('\n[RESULTS]')
    models_to_test = [('fp16 (base)', base_model_path)] + list(zip(QUANT_TYPES, quant_model_paths))
    for label, path in models_to_test:
        t = run_inference(path)
        print(f'{label}: {t:.2f} ms')
