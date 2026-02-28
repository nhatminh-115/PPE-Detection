import os
from pathlib import Path
import mlflow
import torch
from ultralytics import YOLO

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_CHECKPOINT = Path("yolo11m.pt")

def train_underground_ppe():

    device = 0 if torch.cuda.is_available() else "cpu"

    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing local checkpoint: {MODEL_CHECKPOINT.resolve()}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) #Run mlflow ui --host 127.0.0.1 --port 5000 first to start the mlflow server
    mlflow.set_experiment("PPE_Mining_Detection_v1")

    with mlflow.start_run(run_name="YOLO11m_Dark_Augmentation"):
        
        model = YOLO(str(MODEL_CHECKPOINT))
        loaded_ckpt = getattr(model, "ckpt_path", str(MODEL_CHECKPOINT))
        loaded_name = getattr(model, "model_name", str(MODEL_CHECKPOINT))

        print(f"[TRAIN] Model checkpoint: {loaded_ckpt}")
        print(f"[TRAIN] Model name: {loaded_name}")
        print(f"[TRAIN] Device: {device}")

        mlflow.log_param("model_checkpoint", str(loaded_ckpt))
        mlflow.log_param("model_name", str(loaded_name))
        mlflow.log_param("device", str(device))
        
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
            val=True
        )
        
        best_model_path = "runs/detect/underground_ppe/weights/best.pt"
        mlflow.log_artifact(best_model_path, artifact_path="model_weights")

if __name__ == "__main__":
    train_underground_ppe()