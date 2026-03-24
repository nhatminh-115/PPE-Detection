import logging
import os
import torch
import timm
import concurrent.futures
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from src.config import NUM_CLASSES
from src.infrastructure.mlflow import pull_artifact_from_mlflow_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

os.makedirs("model_cache", exist_ok=True)
os.makedirs("violation_crops", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Inference device: {device}")

model1_path = pull_artifact_from_mlflow_run(
    run_id="b52f0641a3fe41899bb4c620fdef053d",
    artifact_path="weights/best.pt",
)
model_stage2_path = pull_artifact_from_mlflow_run(
    run_id="e44391ef94b54ce3b867d46b7ce33ec3",
    artifact_path="weights/best_stage2_effnet.pt",
)
model2_path    = "yolo26l.pt"
model_pose_path = "yolo26m-pose.pt"

model1 = YOLO(model1_path)
model1.predict(
    torch.zeros((1, 3, 1280, 1280)).to(device),
    imgsz=1280,
    device=device,
)
model2     = YOLO(model2_path)
model_pose = YOLO(model_pose_path)

dummy = torch.zeros((1, 3, 640, 640)).to(device)
model2.predict(dummy, device=device, verbose=False)
model_pose.predict(dummy, device=device, verbose=False)


model_stage2 = timm.create_model(
    'tf_efficientnetv2_b0',
    pretrained=False,
    num_classes=NUM_CLASSES,
)
model_stage2.load_state_dict(
    torch.load(model_stage2_path, map_location=device)
)
model_stage2.eval().to(device)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # +1 for pose

app = FastAPI(title="PPE Detection API", version="3.0.0")
app.mount(
    "/violation_crops",
    StaticFiles(directory="violation_crops"),
    name="violation_crops",
)
templates = Jinja2Templates(directory="templates")