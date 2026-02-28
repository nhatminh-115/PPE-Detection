import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from ultralytics import YOLO
from supabase import create_client, Client
from mlflow.artifacts import download_artifacts

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "your_supabase_url_here")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your_supabase_api_key_here")
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Cảnh báo hệ thống: Không thể kết nối Supabase. Lỗi: {e}")
    supabase = None

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/nhatminh-115/PPE-Detection.mlflow"

DAGSHUB_RUN_ID = os.environ.get("DAGSHUB_RUN_ID", "nhập_run_id_vào_đây")
ARTIFACT_PATH = "production_weights/best.pt"

def initialize_model():
    """
    Hàm khởi tạo mô hình động (Dynamic Initialization).
    Sử dụng giao thức MLflow Artifact API để tải trọng số từ Cloud Storage xuống Cache cục bộ.
    """
    try:
        # Xây dựng định danh URI chuẩn của hệ thống MLflow
        artifact_uri = f"runs:/{DAGSHUB_RUN_ID}/{ARTIFACT_PATH}"
        print(f"Hệ thống: Đang thiết lập kênh truyền tải với Dagshub ({artifact_uri})...")
        
        # Hàm download_artifacts yêu cầu môi trường thực thi phải có MLFLOW_TRACKING_USERNAME và PASSWORD
        local_model_path = download_artifacts(artifact_uri=artifact_uri)
        
        print(f"Hệ thống: Khớp nối trọng số thành công. Đang nạp đồ thị tính toán vào bộ nhớ...")
        return YOLO(local_model_path)
    except Exception as e:
        print(f"Lỗi nghiêm trọng: Phân phối mô hình thất bại. Chi tiết: {e}")
        return None

model = initialize_model()

VIOLATION_CLASSES = ['no_helmet', 'no_gloves', 'no_boots', 'no_goggle', 'none']

@app.route('/detect', methods=['POST'])
def detect_ppe():
    if model is None:
        return jsonify({"error": "Dịch vụ gián đoạn: Mô hình suy luận chưa sẵn sàng."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy payload chứa hình ảnh."}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Định dạng dữ liệu ảnh bất hợp lệ."}), 400

    # Thực thi Suy luận (Inference Execution)
    results = model(img, verbose=False)[0]
    violations_detected = []
    
    for box in results.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())
        
        if class_name in VIOLATION_CLASSES:
            violations_detected.append({
                "violation_type": class_name,
                "confidence": round(confidence, 3)
            })

    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Lưu vết Bất đồng bộ gián tiếp
    if violations_detected and supabase:
        try:
            log_data = {
                "violations": violations_detected,
                "status": "Warning"
            }
            supabase.table("ppe_violations").insert(log_data).execute()
        except Exception as e:
            print(f"Lỗi truy xuất CSDL: {e}")

    response_payload = {
        "status": "success",
        "violations_count": len(violations_detected),
        "details": violations_detected,
        "image_base64": img_base64
    }

    return jsonify(response_payload), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)