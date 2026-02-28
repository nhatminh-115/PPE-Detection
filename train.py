import os
from pathlib import Path
import torch
import mlflow
from ultralytics import YOLO
from ultralytics import settings

DAGSHUB_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/nhatminh-115/PPE-Detection.mlflow")
os.environ["MLFLOW_TRACKING_URI"] = DAGSHUB_URI

settings.update({'mlflow': True, 'hub': False})

def train_underground_ppe():
    device = 0 if torch.cuda.is_available() else "cpu"
    model_checkpoint = "yolo11m.pt"

    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_experiment("PPE_Mining_Detection_v1")

    print(f"[SYSTEM] Hardware Accelerator: {device}")
    print(f"[SYSTEM] Tracking Registry: {DAGSHUB_URI}")

    model = YOLO(model_checkpoint)

    results = model.train(
        data="datasets/data.yaml",
        cfg="mine_hyp.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project="runs/detect",
        name="underground_ppe",
        exist_ok=True,
        val=True,
        workers=2
    )
    
    print("[SYSTEM] Pipeline Completed. Weights and metrics are natively synced to Dagshub.")

if __name__ == "__main__":
    train_underground_ppe()