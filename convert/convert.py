import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from rknn.api import RKNN

import pickle

class DummyObject:
    def __init__(self, *args, **kwargs):
        pass

class DummyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f"[MISSING CLASS] {module}.{name}")
        return DummyObject

    def persistent_load(self, pid):
        print(f"[PERSISTENT LOAD] {pid} (type: {type(pid)})")
        # Возвращаем всегда строку
        return "dummy"

with open("v10nfull.pt", "rb") as f:
    up = DummyUnpickler(f)
    up.load()




# === Настройки ===
load_dotenv()

ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT   = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION   = int(os.getenv("ROBOFLOW_VERSION", "1"))  # Default to version 1 if not set

# Проверка обязательных переменных окружения
if not all([ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT]):
    print("Error: Missing required environment variables:")
    if not ROBOFLOW_API_KEY: print("- ROBOFLOW_API_KEY")
    if not ROBOFLOW_WORKSPACE: print("- ROBOFLOW_WORKSPACE")
    if not ROBOFLOW_PROJECT: print("- ROBOFLOW_PROJECT")
    sys.exit(1)

PT_MODEL       = 'v10nfull.pt'
ONNX_MODEL     = 'v10nfull_no_nms.onnx'
INPUT_SIZE     = (640, 640)
TARGET_PLATFORM= 'rk3588'
QUANT_TYPES    = ['fp16', 'int8', 'int4']

DATASET_DIR    = 'calib_images'
TRAIN_TXT      = 'dataset_train.txt'
VALID_TXT      = 'dataset_valid.txt'

# === 0. Roboflow: загрузка и сплит ===
def download_and_split_dataset():
    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    ds      = project.version(ROBOFLOW_VERSION).download("yolov5")
    os.makedirs(DATASET_DIR, exist_ok=True)
    imgs = []
    for root, _, files in os.walk(ds.location):
        for f in files:
            if f.lower().endswith(('.jpg','.png','jpeg')):
                src = os.path.join(root,f)
                dst = os.path.join(DATASET_DIR,f)
                if not os.path.exists(dst):
                    os.system(f'cp "{src}" "{dst}"')
                imgs.append(os.path.abspath(dst))
    train, valid = train_test_split(imgs, train_size=0.8, random_state=42)
    with open(TRAIN_TXT,'w') as ft: ft.write('\n'.join(train))
    with open(VALID_TXT,'w') as fv: fv.write('\n'.join(valid))
    print(f'[INFO] Train={len(train)}  Valid={len(valid)}')

# === 1. Экспорт в ONNX без NMS ===
def export_pt_to_onnx_no_nms(pt_model, onnx_out):
    model = YOLO(pt_model)
    # экспорт без встроенного postprocess (no NMS)
    model.export(
        format="onnx",
        opset=11,
        dynamic=False,
        simplify=True,
        include_postprocess=False,   # <–– отключаем NMS
        imgsz=INPUT_SIZE
    )
    # ulralytics по умолчанию сохранёт как model.onnx в cwd, переименуем:
    os.replace("model.onnx", onnx_out)
    print(f'[INFO] ONNX без NMS сохранён: {onnx_out}')

# === 2. Конвертация ONNX → RKNN с квантованием ===
def convert_onnx_to_rknn(onnx_path, rknn_path, quant):
    rknn = RKNN()
    rknn.config(target_platform=TARGET_PLATFORM,
                quantize_input=(quant!='fp16'),
                quantized_dtype=(None if quant=='fp16' else quant))
    assert rknn.load_onnx(model=onnx_path)==0
    assert rknn.build(do_quantization=(quant!='fp16'), dataset=TRAIN_TXT)==0
    assert rknn.export_rknn(rknn_path)==0
    rknn.release()
    print(f'[INFO] {quant} → {rknn_path}')

# === 3. Инференс + замер времени + точность ===
def evaluate_model(rknn_path):
    rknn = RKNN()
    assert rknn.load_rknn(rknn_path)==0
    assert rknn.init_runtime(target=TARGET_PLATFORM)==0
    with open(VALID_TXT) as f: val = [l.strip() for l in f if l.strip()]
    total_t=0; corr=0; tot=0
    for imgf in val[:50]:
        img = cv2.imread(imgf)
        img = cv2.resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        t0 = time.time()
        out = rknn.inference(inputs=[img])[0]
        total_t += (time.time()-t0)
        pred = int(np.argmax(out))
        lbl  = extract_label_from_filename(imgf)
        corr += (pred==lbl)
        tot  += 1
    rknn.release()
    return (total_t/tot)*1000, corr/tot*100

def extract_label_from_filename(path):
    for part in os.path.basename(path).split('_'):
        if part.split('.')[0].isdigit():
            return int(part.split('.')[0])
    return 0

# === Main ===
if __name__=='__main__':
    download_and_split_dataset()
    export_pt_to_onnx_no_nms(PT_MODEL, ONNX_MODEL)

    paths = []
    for q in QUANT_TYPES:
        out = f'model_{q}.rknn'
        convert_onnx_to_rknn(ONNX_MODEL, out, q)
        paths.append((q,out))

    print('\n[RESULTS]')
    for q,p in paths:
        t,acc = evaluate_model(p)
        print(f'{q}: {t:.2f} ms | Acc: {acc:.2f}%')
