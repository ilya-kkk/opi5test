import os
import time
import cv2
import numpy as np
from rknn.api import RKNN
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
from roboflow import Roboflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# === Configuration ===
load_dotenv()
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT   = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION   = int(os.getenv("ROBOFLOW_VERSION", "1"))

INPUT_SIZE = (640, 640)
NUM_RUNS = 100
DATASET_DIR = 'calib_images'
VALID_TXT = 'dataset_valid.txt'
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

def download_dataset():
    """Download and prepare dataset from Roboflow"""
    print("[INFO] Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    ds = project.version(ROBOFLOW_VERSION).download("yolov5")
    
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
    
    # Use 20% for validation
    _, valid = train_test_split(imgs, train_size=0.8, random_state=42)
    with open(VALID_TXT,'w') as fv: fv.write('\n'.join(valid))
    print(f"[INFO] Dataset prepared. Validation set: {len(valid)} images")
    return valid

def load_model(model_path):
    """Load RKNN model"""
    rknn = RKNN()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f'[ERROR] Load RKNN model failed: {ret}')
        return None

    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588', device_id='npu0')
    if ret != 0:
        print(f'[ERROR] Init runtime environment failed: {ret}')
        return None

    return rknn

def preprocess_image(image_path):
    """Preprocess image for inference"""
    img = cv2.imread(image_path)
    if img is None:
        print(f'[ERROR] Could not read image: {image_path}')
        return None

    img = cv2.resize(img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)

def extract_label_from_filename(path):
    """Extract label from filename"""
    for part in os.path.basename(path).split('_'):
        if part.split('.')[0].isdigit():
            return int(part.split('.')[0])
    return 0

def process_output(output, conf_threshold=CONF_THRESHOLD):
    """Process YOLOv5 output to get predictions"""
    # Reshape output to [num_boxes, num_classes + 5]
    predictions = output[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4:]  # class scores
    
    # Get class with highest confidence
    class_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # Filter by confidence
    mask = class_scores > conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    class_scores = class_scores[mask]
    
    if len(boxes) == 0:
        return None
    
    # Return the class with highest confidence
    best_idx = np.argmax(class_scores)
    return class_ids[best_idx]

def measure_inference_time(model, image, num_runs=NUM_RUNS):
    """Measure inference time"""
    times = []
    
    # Warmup
    print("Warming up model...")
    for _ in range(10):
        model.inference(inputs=[image])
    
    # Measure time
    print(f"Running inference tests ({num_runs} runs)...")
    for _ in tqdm(range(num_runs), desc="Running inference"):
        start = time.time()
        outputs = model.inference(inputs=[image])
        end = time.time()
        times.append((end - start) * 1000)  # convert to ms
    
    return times

def measure_accuracy(model, validation_images):
    """Measure model accuracy on validation set"""
    correct = 0
    total = 0
    
    print("Measuring accuracy on validation set...")
    for img_path in tqdm(validation_images, desc="Evaluating accuracy"):
        img = preprocess_image(img_path)
        if img is None:
            continue
            
        out = model.inference(inputs=[img])[0]
        pred = process_output(out)
        if pred is None:
            continue
            
        lbl = extract_label_from_filename(img_path)
        correct += (pred == lbl)
        total += 1
    
    return (correct / total * 100.0) if total > 0 else 0.0

def test_model(model_path, validation_images):
    """Test model performance"""
    print(f'\n[INFO] Testing model: {os.path.basename(model_path)}')
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return None

    # Get first image for timing test
    test_image = preprocess_image(validation_images[0])
    if test_image is None:
        return None

    # Measure inference time
    times = measure_inference_time(model, test_image)
    
    # Measure accuracy
    accuracy = measure_accuracy(model, validation_images)
    
    # Calculate statistics
    stats = {
        'Model': os.path.basename(model_path),
        'Avg Time (ms)': np.mean(times),
        'Std Dev (ms)': np.std(times),
        'Min Time (ms)': np.min(times),
        'Max Time (ms)': np.max(times),
        'FPS': 1000/np.mean(times),
        'Accuracy (%)': accuracy
    }
    
    # Cleanup
    model.release()
    return stats

def main():
    # List of models to test
    models = [
        'v10nint8256_fp16.rknn',
        'v10nint8256_int8.rknn'
    ]

    # Download and prepare dataset
    validation_images = download_dataset()
    if not validation_images:
        print("[ERROR] Failed to prepare dataset")
        return

    # Test all models
    results = []
    for model_path in models:
        if not os.path.exists(model_path):
            print(f'[WARNING] Model {model_path} not found, skipping...')
            continue
            
        stats = test_model(model_path, validation_images)
        if stats:
            results.append(stats)

    if not results:
        print('[ERROR] No models were successfully tested')
        return

    # Print results
    print('\n[RESULTS]')
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt='.2f'))

    # Save results
    csv_path = 'inference_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n[INFO] Results saved to {csv_path}')

if __name__ == '__main__':
    main()
