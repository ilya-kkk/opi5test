from ultralytics import YOLO
import os
import sys

# Create directory if it doesn't exist
MODEL_DIR = '../1model_to_convert'
OUTPUT_DIR = '../3onnx_to_rknn'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if directory is empty
files = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f))]

if not files:
    print(f"[ERROR] No model files found in {MODEL_DIR}")
    print(f"Please place your .pt model file in the {MODEL_DIR} directory")
    sys.exit(1)

if len(files) > 1:
    print(f"[ERROR] Multiple files found in {MODEL_DIR}")
    print("Please place only one .pt model file in the directory")
    sys.exit(1)

# Get the model file
model_file = os.path.join(MODEL_DIR, files[0])
print(f"[INFO] Converting model: {model_file}")

try:
    # Load and convert model
    yolo = YOLO(model_file)
    
    # Export to ONNX without NMS/TopK
    output_path = yolo.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        opset=12,  # ONNX opset version
        half=False,  # FP32
        int8=False,  # No INT8 quantization
        device='cpu',  # Export on CPU
        verbose=True,
        project=OUTPUT_DIR  # Save to specified directory
    )
    
    print(f"[SUCCESS] Model converted and saved to: {output_path}")

except Exception as e:
    print(f"[ERROR] Failed to convert model: {str(e)}")
    sys.exit(1)

