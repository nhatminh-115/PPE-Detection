import os
import logging
import warnings
import torch
import mlflow
from mlflow.artifacts import download_artifacts
from dotenv import load_dotenv

# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================
load_dotenv()

# --- A. MLflow & MLOps Configuration ---
DAGSHUB_URI = "https://dagshub.com/nhatminh-115/SentinelVision.mlflow"

EXPERIMENT_NAME = "PPE_Stage1_Human_Detection" 
RUN_NAME = "VisDrone_Human_Base_Nano_3"
SOURCE_RUN_ID = "b52f0641a3fe41899bb4c620fdef053d"

PROJECT_ROOT = os.path.abspath(os.getcwd())
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
DATA_YAML_PATH = os.path.join(DATASETS_DIR, "stage1_human.yaml")

MODEL_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "runs",
    "detect",
    "PPE_Stage1_Human_Detection",
    "VisDrone_Human_Base_Nano_2",
    "weights",
    "last.pt",
)
EPOCHS = 100
BATCH_SIZE = -1
IMG_SIZE = 960
WORKERS = 4
PATIENCE = 25

OPTIMIZER = "AdamW"
LR0 = 0.001
LRF = 0.1
WEIGHT_DECAY = 0.0005

MOSAIC = 1.0
MIXUP = 0.0
DEGREES = 5.0
HSV_S = 0.7
COPY_PASTE = 0.3

VERBOSE_MODE = True 


def _silence_ultralytics_noise():
    os.environ["YOLO_VERBOSE"] = "True"

    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", category=UserWarning, module=r"ultralytics(\..*)?")


def _resolve_source_weights():
    if not SOURCE_RUN_ID:
        return MODEL_WEIGHTS

    candidate_artifacts = [
        "production_weights/best.pt",
        "weights/best.pt",
    ]

    for artifact_path in candidate_artifacts:
        try:
            local_path = download_artifacts(artifact_uri=f"runs:/{SOURCE_RUN_ID}/{artifact_path}")
            if os.path.exists(local_path):
                print(f"[SYSTEM] Đã nạp checkpoint từ Run ID {SOURCE_RUN_ID}: {artifact_path}")
                return local_path
        except Exception:
            continue

    raise FileNotFoundError(
        f"[FATAL ERROR] Không tải được best weight từ Run ID {SOURCE_RUN_ID}. "
        "Đã thử: production_weights/best.pt, weights/best.pt"
    )

# ==============================================================================
# 2. SYSTEM INITIALIZATION & TRAINING PIPELINE
# ==============================================================================
def train_stage1():
    _silence_ultralytics_noise()
    from ultralytics import YOLO, settings

    os.environ["MLFLOW_TRACKING_URI"] = DAGSHUB_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    mlflow.set_tracking_uri(DAGSHUB_URI)

    settings.update({
        'mlflow': True, 
        'hub': False, 
        'datasets_dir': DATASETS_DIR
    })

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Hardware Accelerator: {device}")
    print(f"[SYSTEM] Absolute Dataset Path: {DATA_YAML_PATH}")

    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"[FATAL ERROR] Không tìm thấy file {DATA_YAML_PATH}.")

    source_weights = _resolve_source_weights()

    if not os.path.exists(source_weights):
        raise FileNotFoundError(f"[FATAL ERROR] Không tìm thấy checkpoint {source_weights}.")

    model = YOLO(source_weights)

    model.train(
        data=DATA_YAML_PATH,
        project=EXPERIMENT_NAME, 
        name=RUN_NAME,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        mosaic=MOSAIC,
        mixup=MIXUP,
        degrees=DEGREES,
        hsv_s=HSV_S,
        exist_ok=True,
        val=True,
        workers=WORKERS,
        save_period=10,
        patience=PATIENCE,
        verbose=VERBOSE_MODE,
        copy_paste=COPY_PASTE
    )
    
    # Thư mục local bây giờ sẽ là: D:\Work\Project PPE\PPE-Detection\PPE_Stage1_Human_Detection\VisDrone_Human_Base_Nano\weights
    best_model_path = os.path.join(PROJECT_ROOT, EXPERIMENT_NAME, RUN_NAME, "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        print(f"\n[SYSTEM] Model Converged! Đang tải {best_model_path} lên Registry...")
        mlflow.set_tracking_uri(DAGSHUB_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"{RUN_NAME}_Artifacts"):
            mlflow.log_artifact(best_model_path, artifact_path="production_weights")
        print("[SUCCESS] Vận chuyển thành công!")

if __name__ == "__main__":
    train_stage1()