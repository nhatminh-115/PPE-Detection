import logging
import os
import warnings
import io
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import torch
import open_clip
import timm
import concurrent.futures
from torchvision import transforms
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from src.config import (
    SIGLIP_MODEL_NAME, SIGLIP_PRETRAINED, SIGLIP_PROMPTS,
    IMG_SIZE, NUM_CLASSES,
    EFFNET_RUN_ID, EFFNET_ARTIFACT_PATH, EFFNET_LOCAL_FALLBACK_PATH,
    MLFLOW_RUN_ID, MLFLOW_ARTIFACT_PATH,
    MODEL2_PATH, MODEL_POSE_PATH,
    INFERENCE_THREAD_WORKERS,
)
from src.infrastructure.mlflow import pull_artifact_from_mlflow_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)


class _StartupNoiseFilter(logging.Filter):
    """Drop known noisy INFO logs from tokenizer/model startup."""

    _needles = (
        "HFTokenizer",
        "Parsing tokenizer identifier",
        "Parsing model identifier",
        "Attempting to load config from built-in",
        "Loaded built-in",
        "Instantiating model architecture",
        "Loading full pretrained weights from",
        "Final image preprocessing configuration set",
        "creation process complete",
        "HTTP Request: HEAD https://huggingface.co",
        "HTTP Request: GET https://huggingface.co",
        "unauthenticated requests to the HF Hub",
    )

    def filter(self, record):
        msg = record.getMessage()
        if any(needle in msg for needle in self._needles):
            return False
        return True


if os.getenv("MUTE_THIRD_PARTY_STARTUP_LOGS", "1") == "1":
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING", "1")
    warnings.filterwarnings(
        "ignore",
        message=".*[Uu]nauthenticated requests to the HF Hub.*",
    )
    quiet_level = getattr(logging, os.getenv("THIRD_PARTY_LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    for noisy_logger in (
        "open_clip",
        "open_clip.factory",
        "open_clip.tokenizer",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(noisy_logger).setLevel(quiet_level)
    logging.getLogger().addFilter(_StartupNoiseFilter())

MUTE_STARTUP_NOISE = os.getenv("MUTE_THIRD_PARTY_STARTUP_LOGS", "1") == "1"


def _quiet_startup_contexts():
    if MUTE_STARTUP_NOISE:
        return redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())
    return nullcontext(), nullcontext()

logger = logging.getLogger(__name__)

os.makedirs("model_cache", exist_ok=True)
os.makedirs("violation_crops", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Inference device: {device}")

model1_path     = pull_artifact_from_mlflow_run(MLFLOW_RUN_ID, MLFLOW_ARTIFACT_PATH)
model2_path     = MODEL2_PATH
model_pose_path = MODEL_POSE_PATH

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
# SigLIP model + text features
# ---------------------------------------------------------------------------

logger.info(f"Loading SigLIP {SIGLIP_MODEL_NAME} ({SIGLIP_PRETRAINED})...")
stdout_ctx, stderr_ctx = _quiet_startup_contexts()
with stdout_ctx, stderr_ctx:
    siglip_model, _, siglip_preprocess = open_clip.create_model_and_transforms(
        SIGLIP_MODEL_NAME, pretrained=SIGLIP_PRETRAINED,
    )
siglip_model.eval().to(device)

# Pre-compute text embeddings (once at startup)
stdout_ctx, stderr_ctx = _quiet_startup_contexts()
with stdout_ctx, stderr_ctx:
    tokenizer = open_clip.get_tokenizer(SIGLIP_MODEL_NAME)
siglip_text_features = {}

with torch.no_grad():
    for category, prompts in SIGLIP_PROMPTS.items():
        pos_tokens = tokenizer(prompts['positive']).to(device)
        neg_tokens = tokenizer(prompts['negative']).to(device)
        pos_feat = siglip_model.encode_text(pos_tokens).mean(dim=0)
        neg_feat = siglip_model.encode_text(neg_tokens).mean(dim=0)
        pos_feat = pos_feat / pos_feat.norm()
        neg_feat = neg_feat / neg_feat.norm()
        siglip_text_features[category] = torch.stack([pos_feat, neg_feat])

# Warmup
with torch.no_grad():
    siglip_model.encode_image(torch.zeros(1, 3, 224, 224).to(device))
logger.info("SigLIP model ready.")

effnet_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_effnet_bundle():
    effnet_mlflow_path = pull_artifact_from_mlflow_run(
        run_id=EFFNET_RUN_ID,
        artifact_path=EFFNET_ARTIFACT_PATH,
    )
    candidates = [
        effnet_mlflow_path,
        EFFNET_LOCAL_FALLBACK_PATH,
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

    logger.warning("No compatible EffNet stage-2 weights found. Router will fall back to SigLIP.")
    return None


# Bundle for classifier
model_stage2 = {
    "siglip": {
        "model": siglip_model,
        "preprocess": siglip_preprocess,
        "text_features": siglip_text_features,
    },
    "effnet": _load_effnet_bundle(),
}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=INFERENCE_THREAD_WORKERS)

app = FastAPI(title="PPE Detection API", version="3.0.0")
app.mount(
    "/violation_crops",
    StaticFiles(directory="violation_crops"),
    name="violation_crops",
)
templates = Jinja2Templates(directory="templates")