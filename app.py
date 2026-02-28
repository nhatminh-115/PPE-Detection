import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from ultralytics import YOLO
from supabase import create_client, Client
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "your_supabase_url_here")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your_supabase_api_key_here")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"[SYSTEM WARNING] Supabase connection failed. Execution will proceed without logging. Details: {e}")
    supabase = None

# 2. Tracking Server Configuration
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/nhatminh-115/PPE-Detection.mlflow"

def initialize_model(experiment_name="PPE_Mining_Production"):
    """
    Hàm khởi tạo mô hình động tự động hoàn toàn (Fully Automated Dynamic Initialization).
    Thực hiện truy vấn Metadata trên Model Registry và nạp trọng số mới nhất vào RAM.
    """
    try:
        print(f"[SYSTEM] Querying Model Registry for Experiment: '{experiment_name}'...")
        client = MlflowClient()
        
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment partition not found: {experiment_name}")
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("No historical training runs found in this Experiment.")
            
        latest_run_id = runs[0].info.run_id
        print(f"[SYSTEM] Discovered latest Run ID: {latest_run_id}")
        
        artifact_path = "production_weights/best.pt"
        artifact_uri = f"runs:/{latest_run_id}/{artifact_path}"
        
        print(f"[SYSTEM] Initiating artifact download stream from {artifact_uri}...")
        local_model_path = download_artifacts(artifact_uri=artifact_uri)
        
        print(f"[SYSTEM] Computational graph loaded successfully. Engine is ready.")
        return YOLO(local_model_path)
        
    except Exception as e:
        print(f"[SYSTEM ERROR] Model synchronization failed. Details: {e}")
        return None

model = initialize_model(experiment_name="PPE_Mining_Production")

RAW_MODEL_CLASSES = ['no_helmet', 'no_gloves', 'no_boots', 'no_goggle', 'none']

@app.route('/detect', methods=['POST'])
def detect_ppe():
    if model is None:
        return jsonify({"error": "Service Unavailable: Inference engine is not initialized."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "Bad Request: Image payload is missing."}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Bad Request: Invalid image format."}), 400

    results = model(img, verbose=False)[0]
    violations_detected = []
    
    for box in results.boxes:
        class_id = int(box.cls[0].item())
        raw_class_name = model.names[class_id]
        confidence = float(box.conf[0].item())
        
        if raw_class_name in RAW_MODEL_CLASSES:
            business_label = 'none_vest' if raw_class_name == 'none' else raw_class_name
            
            violations_detected.append({
                "violation_type": business_label,
                "confidence": round(confidence, 3)
            })

    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    if violations_detected and supabase:
        try:
            log_data = {
                "violations": violations_detected,
                "status": "Warning"
            }
            supabase.table("ppe_violations").insert(log_data).execute()
        except Exception as e:
            print(f"[DATABASE ERROR] Payload insertion failed: {e}")

    response_payload = {
        "status": "success",
        "violations_count": len(violations_detected),
        "details": violations_detected,
        "image_base64": img_base64
    }

    return jsonify(response_payload), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)