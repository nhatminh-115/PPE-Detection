import logging
import os
import warnings
import io
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import torch
import open_clip
import timm
import concurrent.futures
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from src.config import (
    SIGLIP_MODEL_NAME, SIGLIP_PRETRAINED, SIGLIP_PROMPTS,
    IMG_SIZE, NUM_CLASSES,
    EFFNET_RUN_ID, EFFNET_STABLE_RUN_ID,
    EFFNET_ONNX_ARTIFACT_PATH, EFFNET_ONNX_LOCAL_FALLBACK,
    EFFNET_ONNX_INT8_FALLBACK, EFFNET_PT_ARTIFACT_PATH, EFFNET_PT_LOCAL_FALLBACK,
    EFFNET_USE_ONNX, EFFNET_PREFER_RECENT_PT,
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


class _AccessLogNoiseFilter(logging.Filter):
    """Reduce repetitive GET access logs from polling/stream routes and known missing assets."""

    _quiet_get_prefixes = tuple(
        p.strip()
        for p in os.getenv(
            "QUIET_GET_PATH_PREFIXES",
            "/video_feed,/api/violations,/api/cameras,/api/labels,/snapshot,/violation_crops,/favicon.ico",
        ).split(",")
        if p.strip()
    )

    # 404s on these prefixes are expected (e.g. crop images flushed from disk).
    _quiet_404_prefixes = ("/violation_crops/",)

    def filter(self, record):
        if os.getenv("QUIET_GET_ACCESS_LOGS", "1") != "1":
            return True

        args = getattr(record, "args", None)
        if not isinstance(args, tuple) or len(args) < 5:
            return True

        method = str(args[1]).upper()
        path = str(args[2]).split("?", 1)[0]
        try:
            status_code = int(args[4])
        except (TypeError, ValueError):
            status_code = None

        if method != "GET":
            return True

        # Suppress expected 404s for known missing-asset paths
        if status_code == 404 and any(path.startswith(p) for p in self._quiet_404_prefixes):
            return False

        if status_code is not None and status_code >= 400:
            return True

        return not any(path.startswith(prefix) for prefix in self._quiet_get_prefixes)


def _configure_access_logging() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if any(isinstance(f, _AccessLogNoiseFilter) for f in access_logger.filters):
        return
    access_logger.addFilter(_AccessLogNoiseFilter())


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

_configure_access_logging()

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


def _load_effnet_pytorch():
    """Load PyTorch fallback from local file only (no remote pull)."""
    if EFFNET_PT_LOCAL_FALLBACK and os.path.exists(EFFNET_PT_LOCAL_FALLBACK):
        try:
            logger.info(f"Loading EffNet PyTorch from {EFFNET_PT_LOCAL_FALLBACK}...")
            effnet = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=NUM_CLASSES)
            state = torch.load(EFFNET_PT_LOCAL_FALLBACK, map_location=device)

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state.items()
                }

            effnet.load_state_dict(state, strict=False)
            effnet.eval().to(device)
            with torch.no_grad():
                effnet(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device))

            logger.info("✓ EffNet PyTorch model ready (local).")
            return {
                "model": effnet,
                "preprocess": effnet_preprocess,
                "weights_path": EFFNET_PT_LOCAL_FALLBACK,
                "format": "pytorch",
            }
        except Exception as ex:
            logger.warning(f"EffNet PyTorch load failed for {EFFNET_PT_LOCAL_FALLBACK}: {ex}")

    return None


def _load_effnet_bundle():
    """
    Load EfficientNet stage-2 classifier with smart strategy.
    
    If PREFER_RECENT_PT=True (default):
      1. Try recent PT from EFFNET_RUN_ID (likely new retrain) → use immediately
      2. Try stable ONNX from EFFNET_STABLE_RUN_ID → proven production quality
      3. Local ONNX cache (FP32/INT8)
      4. Fallback to PT
      
    If PREFER_RECENT_PT=False:
      1. Try stable ONNX first (proven quality)
      2. Then recent PT (retrain)
      3. Local ONNX
      4. Fallback to PT
      
    This balances:
    - New retrain adoption (recent PT available immediately after drift trigger)
    - Production stability (stable ONNX always available as fallback)
    - Performance (ONNX 6x faster than PT)
    """
    if EFFNET_PREFER_RECENT_PT:
        # Strategy 1: Prefer recent PT retrain (new models used immediately)
        logger.debug(f"Loading EffNet: Priority = RECENT_PT > STABLE_ONNX > LOCAL > FALLBACK_PT")
        
        # Try recent PT from retrain
        bundle = _load_effnet_pytorch_from_run(EFFNET_RUN_ID)
        if bundle is not None:
            logger.info("✓ Loaded recent PT from retrain run (EffNet latest)")
            return bundle
        
        # Try stable ONNX (proven, trusted)
        bundle = _load_effnet_onnx_from_run(EFFNET_STABLE_RUN_ID)
        if bundle is not None:
            logger.info("✓ Loaded stable ONNX from verified run (EffNet stable)")
            return bundle
        
        # Try local ONNX (pre-cached)
        bundle = _load_effnet_onnx_local()
        if bundle is not None:
            logger.info("✓ Loaded ONNX from local cache")
            return bundle
        
        # Last resort: PT fallback
        bundle = _load_effnet_pytorch()
        if bundle is not None:
            logger.info("⚠ Loaded PyTorch fallback (ONNX unavailable)")
            return bundle
    else:
        # Strategy 2: Prefer stable ONNX (safer, proven)
        logger.debug(f"Loading EffNet: Priority = STABLE_ONNX > RECENT_PT > LOCAL > FALLBACK_PT")
        
        # Try stable ONNX first
        bundle = _load_effnet_onnx_from_run(EFFNET_STABLE_RUN_ID)
        if bundle is not None:
            logger.info("✓ Loaded stable ONNX from verified run")
            return bundle
        
        # Try recent PT (newer, but untested)
        bundle = _load_effnet_pytorch_from_run(EFFNET_RUN_ID)
        if bundle is not None:
            logger.info("○ Loaded recent PT (stable ONNX unavailable)")
            return bundle
        
        # Try local cache
        bundle = _load_effnet_onnx_local()
        if bundle is not None:
            logger.info("✓ Loaded ONNX from local cache")
            return bundle
        
        # Fallback
        bundle = _load_effnet_pytorch()
        if bundle is not None:
            logger.info("⚠ Loaded PyTorch fallback")
            return bundle
    
    logger.warning("✗ No compatible EffNet stage-2 weights found. Router will fall back to SigLIP.")
    return None


def _load_effnet_onnx_local():
    """Load ONNX from local cache (no remote pull)."""
    for path in [EFFNET_ONNX_LOCAL_FALLBACK, EFFNET_ONNX_INT8_FALLBACK]:
        if path and os.path.exists(path):
            try:
                logger.info(f"Loading EffNet ONNX from {path}...")
                session = ort.InferenceSession(
                    path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    sess_options=ort.SessionOptions(),
                )
                return {
                    "model": session,
                    "preprocess": effnet_preprocess,
                    "weights_path": path,
                    "format": "onnx",
                }
            except Exception as ex:
                logger.warning(f"Failed to load {path}: {ex}")
    return None


def _load_effnet_onnx_from_run(run_id: str):
    """Load ONNX from specific MLflow run (with remote pull)."""
    try:
        onnx_mlflow_path = pull_artifact_from_mlflow_run(
            run_id=run_id,
            artifact_path=EFFNET_ONNX_ARTIFACT_PATH,
        )
        if onnx_mlflow_path and os.path.exists(onnx_mlflow_path):
            logger.info(f"Loading EffNet ONNX from MLflow run {run_id}...")
            session = ort.InferenceSession(
                onnx_mlflow_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                sess_options=ort.SessionOptions(),
            )
            return {
                "model": session,
                "preprocess": effnet_preprocess,
                "weights_path": onnx_mlflow_path,
                "format": "onnx",
                "source_run_id": run_id,
            }
    except Exception as ex:
        logger.debug(f"Failed to load ONNX from run {run_id}: {ex}")
    return None


def _load_effnet_pytorch_from_run(run_id: str):
    """Load PyTorch from specific MLflow run (with remote pull)."""
    try:
        effnet_mlflow_path = pull_artifact_from_mlflow_run(
            run_id=run_id,
            artifact_path=EFFNET_PT_ARTIFACT_PATH,
        )
        if effnet_mlflow_path and os.path.exists(effnet_mlflow_path):
            logger.info(f"Loading EffNet PyTorch from MLflow run {run_id}...")
            effnet = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=NUM_CLASSES)
            state = torch.load(effnet_mlflow_path, map_location=device)

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

            effnet.load_state_dict(state, strict=False)
            effnet.eval().to(device)
            with torch.no_grad():
                effnet(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device))

            return {
                "model": effnet,
                "preprocess": effnet_preprocess,
                "weights_path": effnet_mlflow_path,
                "format": "pytorch",
                "source_run_id": run_id,
            }
    except Exception as ex:
        logger.debug(f"Failed to load PyTorch from run {run_id}: {ex}")
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