from ultralytics import YOLO
import os
import sys
import onnx
import onnx_graphsurgeon as gs

# === Paths ===
MODEL_DIR = '../1model_to_convert'
OUTPUT_DIR = '../3onnx_to_rknn'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Check model ===
files = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f))]
if not files:
    print(f"[ERROR] No model files found in {MODEL_DIR}")
    print(f"Please place your .pt model file in the {MODEL_DIR} directory")
    sys.exit(1)

if len(files) > 1:
    print(f"[ERROR] Multiple files found in {MODEL_DIR}")
    print("Please place only one .pt model file in the directory")
    sys.exit(1)

model_file = os.path.join(MODEL_DIR, files[0])
print(f"[INFO] Converting model: {model_file}")

try:
    # === Step 1: Convert to ONNX ===
    yolo = YOLO(model_file)
    onnx_path = yolo.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        opset=12,
        half=False,
        int8=False,
        device='cpu',
        verbose=True,
        project=OUTPUT_DIR,
        name='converted_model'  # Optional: name of subdir
    )

    print(f"[INFO] Exported ONNX model: {onnx_path}")

    # === Step 2: Remove TopK and NMS ===
    print("[INFO] Cleaning up ONNX model (removing TopK and NMS nodes)...")
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)

    original_node_count = len(graph.nodes)
    graph.nodes = [n for n in graph.nodes if n.op not in ['TopK', 'NonMaxSuppression']]
    removed = original_node_count - len(graph.nodes)

    graph.cleanup().toposort()

    cleaned_path = os.path.join(OUTPUT_DIR, "yolo_no_topk.onnx")
    onnx.save(gs.export_onnx(graph), cleaned_path)

    print(f"[SUCCESS] Cleaned ONNX saved to: {cleaned_path} (removed {removed} nodes)")

except Exception as e:
    print(f"[ERROR] Failed: {str(e)}")
    sys.exit(1)
