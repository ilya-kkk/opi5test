import os
import sys
import time
import cv2
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from rknn.api import RKNN
import onnx
import onnx_graphsurgeon as gs

# === Настройки ===
load_dotenv()
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT   = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION   = int(os.getenv("ROBOFLOW_VERSION", "1"))
for var in ("ROBOFLOW_API_KEY","ROBOFLOW_WORKSPACE","ROBOFLOW_PROJECT"):
    if not os.getenv(var):
        print(f"Error: Missing {var}"); sys.exit(1)

ONNX_MODEL      = 'yolo_no_topk.onnx'
INPUT_SIZE      = (640, 640)
TARGET_PLATFORM = 'rk3588'
QUANT_TYPES     = ['fp16', 'int8']

DATASET_DIR = 'calib_images'
TRAIN_TXT   = 'dataset_train.txt'
VALID_TXT   = 'dataset_valid.txt'

def clean_onnx_model(onnx_path):
    """Remove TopK and NMS nodes from ONNX model"""
    print(f"[INFO] Cleaning ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)

    # Find and remove TopK and NMS nodes
    nodes_to_remove = []
    for node in graph.nodes:
        if node.op in ['TopK', 'NonMaxSuppression']:
            nodes_to_remove.append(node)
            # Connect input of TopK/NMS to its output
            for output in node.outputs:
                for consumer in output.outputs:
                    consumer.inputs = [node.inputs[0]]  # Use first input of TopK/NMS

    # Remove the nodes
    for node in nodes_to_remove:
        graph.nodes.remove(node)

    # Clean up and save
    graph.cleanup().toposort()
    cleaned_path = onnx_path.replace('.onnx', '_cleaned.onnx')
    onnx.save(gs.export_onnx(graph), cleaned_path)
    print(f"[INFO] Cleaned model saved to: {cleaned_path}")
    return cleaned_path

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

def convert_onnx_to_rknn(onnx_path, rknn_path, quant):
    print(f"[INFO] Converting {onnx_path} → {quant}")
    rknn = RKNN()

    # Updated configurations for RK3588
    if quant == 'fp16':
        cfg = {
            'target_platform': TARGET_PLATFORM,
            'float_dtype': 'float16',
        }
        do_quant = False

    elif quant == 'int8':
        cfg = {
            'target_platform': TARGET_PLATFORM,
            'quantized_dtype': 'w8a8',
        }
        do_quant = True

    else:
        raise ValueError(f"Unsupported quant type: {quant}")

    # Apply config
    rknn.config(**cfg)

    # Conversion
    assert rknn.load_onnx(model=onnx_path) == 0
    assert rknn.build(do_quantization=do_quant, dataset=TRAIN_TXT) == 0
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
    # Clean ONNX model first
    cleaned_onnx = clean_onnx_model(ONNX_MODEL)
    
    # Download dataset
    download_and_split_dataset()

    # Convert to RKNN
    results = []
    for q in QUANT_TYPES:
        out = f"v10nint8256_{q}.rknn"
        convert_onnx_to_rknn(cleaned_onnx, out, q)
        results.append((q, out))

    print("\n[RESULTS]")
    for q, path in results:
        t, acc = evaluate_model(path)
        print(f"{q:6} | {t:8.2f} ms | {acc:6.2f}%")
