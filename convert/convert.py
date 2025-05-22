import os
import sys
import time
import cv2
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from rknn.api import RKNN

# === Настройки ===
load_dotenv()
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT   = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION   = int(os.getenv("ROBOFLOW_VERSION", "1"))
for var in ("ROBOFLOW_API_KEY","ROBOFLOW_WORKSPACE","ROBOFLOW_PROJECT"):
    if not os.getenv(var):
        print(f"Error: Missing {var}"); sys.exit(1)

ONNX_MODEL      = 'v10nint8256.onnx'
INPUT_SIZE      = (640, 640)
TARGET_PLATFORM = 'rk3588'
QUANT_TYPES     = ['fp16', 'int8', 'int4']

DATASET_DIR = 'calib_images'
TRAIN_TXT   = 'dataset_train.txt'
VALID_TXT   = 'dataset_valid.txt'

# === 0. Roboflow: загрузка и сплит ===
def download_and_split_dataset():
    print("[INFO] Downloading dataset from Roboflow...")
    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    ds      = project.version(ROBOFLOW_VERSION).download("yolov5")
    os.makedirs(DATASET_DIR, exist_ok=True)
    imgs = []
    for root, _, files in os.walk(ds.location):
        for f in files:
            if f.lower().endswith(('.jpg','jpeg','png')):
                src = os.path.join(root,f)
                dst = os.path.join(DATASET_DIR,f)
                if not os.path.exists(dst):
                    os.system(f'cp "{src}" "{dst}"')
                imgs.append(os.path.abspath(dst))
    train, valid = train_test_split(imgs, train_size=0.8, random_state=42)
    with open(TRAIN_TXT,'w') as ft: ft.write('\n'.join(train))
    with open(VALID_TXT,'w') as fv: fv.write('\n'.join(valid))
    print(f"[INFO] Split done. Train={len(train)} Valid={len(valid)}")

# === 1. Конвертация ONNX → RKNN с квантованием ===
def convert_onnx_to_rknn(onnx_path, rknn_path, quant):
    print(f"[INFO] Converting {onnx_path} → {quant}")
    rknn = RKNN()

    # Настройка квантования
    quant_settings = {
        'fp16':   {'quantize_input': False, 'quantized_dtype': 'w16a16i'},
        'int8':   {'quantize_input': True,  'quantized_dtype': 'w8a8'},
        'int4':   {'quantize_input': True,  'quantized_dtype': 'w4a16'},
    }

    if quant not in quant_settings:
        raise ValueError(f"Unsupported quant type: {quant}")

    rknn.config(
        target_platform=TARGET_PLATFORM,
        quantize_input=quant_settings[quant]['quantize_input'],
        quantized_dtype=quant_settings[quant]['quantized_dtype']
    )

    assert rknn.load_onnx(model=onnx_path) == 0
    assert rknn.build(do_quantization=quant != 'fp16', dataset=TRAIN_TXT) == 0
    assert rknn.export_rknn(rknn_path) == 0
    rknn.release()
    print(f"[INFO] Exported RKNN model: {rknn_path}")

# === 2. Инференс + замер времени + точность ===
def evaluate_model(rknn_path):
    print(f"[INFO] Evaluating {rknn_path}")
    rknn = RKNN()
    assert rknn.load_rknn(rknn_path) == 0
    assert rknn.init_runtime(target=TARGET_PLATFORM) == 0

    with open(VALID_TXT) as f:
        val_list = [l.strip() for l in f if l.strip()]

    total_time = 0.0
    correct = 0
    total = 0

    for imgf in val_list[:50]:
        img = cv2.imread(imgf)
        img = cv2.resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        t0 = time.time()
        out = rknn.inference(inputs=[img])[0]
        total_time += (time.time() - t0)

        pred = int(np.argmax(out))
        lbl  = extract_label_from_filename(imgf)
        correct += (pred == lbl)
        total += 1

    rknn.release()
    if total == 0:
        return 0.0, 0.0
    return (total_time / total) * 1000, correct / total * 100.0

def extract_label_from_filename(path):
    for part in os.path.basename(path).split('_'):
        if part.split('.')[0].isdigit():
            return int(part.split('.')[0])
    return 0

# === Main ===
if __name__ == "__main__":
    download_and_split_dataset()

    results = []
    for q in QUANT_TYPES:
        out = f"v10nint8256_{q}.rknn"
        convert_onnx_to_rknn(ONNX_MODEL, out, q)
        results.append((q, out))

    print("\n[RESULTS]")
    for q, path in results:
        t, acc = evaluate_model(path)
        print(f"{q:6} | {t:8.2f} ms | {acc:6.2f}%")
