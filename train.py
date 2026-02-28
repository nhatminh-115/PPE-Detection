import os
from dotenv import load_dotenv
import torch
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
from ultralytics import settings

load_dotenv()

DAGSHUB_URI = "https://dagshub.com/nhatminh-115/PPE-Detection.mlflow"
os.environ["MLFLOW_TRACKING_URI"] = DAGSHUB_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

settings.update({'mlflow': True, 'hub': False})

def fetch_latest_domain_weights(experiment_name="PPE_Mining_Production"):
    try:
        print(f"[SYSTEM] Establishing secure channel to Registry: '{experiment_name}'...")
        client = MlflowClient(tracking_uri=DAGSHUB_URI)
        
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return "yolo11m.pt"
            
        # Nâng cấp kiến trúc: Ép buộc hệ thống chỉ lấy các Run đã hoàn thành trọn vẹn
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            return "yolo11m.pt"
            
        latest_run_id = runs[0].info.run_id
        
        artifact_dir_path = "production_weights" 
        
        print(f"[SYSTEM] Extracting domain weights from Run ID [{latest_run_id}]...")
        
        local_dir = client.download_artifacts(run_id=latest_run_id, path=artifact_dir_path)
        local_model_path = os.path.join(local_dir, "best.pt")
        
        if os.path.exists(local_model_path):
            return local_model_path
        else:
            raise FileNotFoundError("Trọng số không tồn tại.")
            
    except Exception as e:
        print(f"[SYSTEM ERROR] Connection interrupted: {e}")
        return "yolo11m.pt"

def train_underground_ppe():
    device = 0 if torch.cuda.is_available() else "cpu"
    mlflow.set_tracking_uri(DAGSHUB_URI)
    
    mlflow.set_experiment("PPE_Mining_Production")
    
    model_checkpoint = fetch_latest_domain_weights("PPE_Mining_Production")
    model = YOLO(model_checkpoint)

    print(f"[SYSTEM] Hardware Accelerator: {device}")
    print(f"[SYSTEM] Computational Graph initialized from: {model_checkpoint}")

    with mlflow.start_run(run_name="Underground_Finetuning_v2") as run:
        print(f"[SYSTEM] Active Telemetry Session: {run.info.run_id}")
        
        mlflow.autolog(log_models=False) 

        model.train(
            data="datasets/data.yaml",
            cfg="mine_hyp.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            device=device,
            project="runs/detect",
            name="underground_ppe_finetune",
            exist_ok=True,
            val=True,
            workers=2
        )
        
        best_model_path = "runs/detect/underground_ppe_finetune/weights/best.pt"
        if os.path.exists(best_model_path):
            print("[SYSTEM] Uploading converged weights to Production Directory...")
            mlflow.log_artifact(best_model_path, artifact_path="production_weights")

if __name__ == "__main__":
    train_underground_ppe()