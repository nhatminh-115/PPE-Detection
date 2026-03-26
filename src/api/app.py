import logging
import os
import glob
import torch
import open_clip
import timm
import concurrent.futures
from torchvision import transforms
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from src.config import CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_PROMPTS, IMG_SIZE, NUM_CLASSES
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


# ---------------------------------------------------------------------------
# CLIP model + text features
# ---------------------------------------------------------------------------

logger.info(f"Loading CLIP {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED,
)
clip_model.eval().to(device)

# Pre-compute text embeddings (once at startup)
tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
clip_text_features = {}

with torch.no_grad():
    for category, prompts in CLIP_PROMPTS.items():
        pos_tokens = tokenizer(prompts['positive']).to(device)
        neg_tokens = tokenizer(prompts['negative']).to(device)
        pos_feat = clip_model.encode_text(pos_tokens).mean(dim=0)
        neg_feat = clip_model.encode_text(neg_tokens).mean(dim=0)
        pos_feat = pos_feat / pos_feat.norm()
        neg_feat = neg_feat / neg_feat.norm()
        clip_text_features[category] = torch.stack([pos_feat, neg_feat])

# Warmup
with torch.no_grad():
    clip_model.encode_image(torch.zeros(1, 3, 224, 224).to(device))
logger.info("CLIP model ready.")

effnet_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_effnet_bundle():
    candidates = [
        "best_stage2_effnet.pt",
        "best_stage2_effnet_attention.pt",
        *sorted(glob.glob("model_cache/*best_stage2_effnet.pt")),
        *sorted(glob.glob("model_cache/*best_stage2_effnet_attention.pt")),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            logger.info(f"Loading EffNet stage-2 weights from {path}...")
            effnet = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=NUM_CLASSES)
            state = torch.load(path, map_location=device)

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state.items()
                }

            missing, unexpected = effnet.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"EffNet missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"EffNet unexpected keys: {len(unexpected)}")

            effnet.eval().to(device)
            with torch.no_grad():
                effnet(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device))

            logger.info("EffNet model ready.")
            return {
                "model": effnet,
                "preprocess": effnet_preprocess,
                "weights_path": path,
            }
        except Exception as ex:
            logger.warning(f"EffNet load failed for {path}: {ex}")

    logger.warning("No compatible EffNet stage-2 weights found. Router will fall back to CLIP.")
    return None


# Bundle for classifier
model_stage2 = {
    "clip": {
        "model": clip_model,
        "preprocess": clip_preprocess,
        "text_features": clip_text_features,
    },
    "effnet": _load_effnet_bundle(),
}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

app = FastAPI(title="PPE Detection API", version="3.0.0")
app.mount(
    "/violation_crops",
    StaticFiles(directory="violation_crops"),
    name="violation_crops",
)
templates = Jinja2Templates(directory="templates")