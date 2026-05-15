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
try:
    import onnxruntime as ort
except ImportError:
    ort = None
from torchvision import transforms
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from src.config import (
    SIGLIP_MODEL_NAME, SIGLIP_PRETRAINED, SIGLIP_PROMPTS,
    SIGLIP_ONNX_ENABLED, SIGLIP_ONNX_EXPORT_ON_STARTUP, SIGLIP_ONNX_PATH,
    SIGLIP_ONNX_HF_REPO,
    IMG_SIZE, NUM_CLASSES,
    EFFNET_RUN_ID, EFFNET_STABLE_RUN_ID, EFFNET_EXPERIMENT_NAME,
    EFFNET_ONNX_ARTIFACT_PATH, EFFNET_ONNX_LOCAL_FALLBACK,
    EFFNET_ONNX_INT8_FALLBACK, EFFNET_PT_ARTIFACT_PATH, EFFNET_PT_LOCAL_FALLBACK,
    EFFNET_USE_ONNX, EFFNET_PREFER_RECENT_PT,
    MLFLOW_RUN_ID, MLFLOW_ARTIFACT_PATH,
    MODEL2_PATH, MODEL_POSE_PATH,
    INFERENCE_THREAD_WORKERS,
)
from src.infrastructure.mlflow import pull_artifact_from_mlflow_run, find_latest_effnet_run_id

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


def _run_quiet_startup(step_name, fn):
    """Run noisy startup steps with optional output suppression and explicit error logging."""
    if not MUTE_STARTUP_NOISE:
        return fn()

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            return fn()
    except Exception:
        captured_stdout = stdout_buffer.getvalue().strip()
        captured_stderr = stderr_buffer.getvalue().strip()
        if captured_stdout:
            logger.error("%s stdout before failure:\n%s", step_name, captured_stdout)
        if captured_stderr:
            logger.error("%s stderr before failure:\n%s", step_name, captured_stderr)
        logger.exception("%s failed during startup.", step_name)
        raise

logger = logging.getLogger(__name__)

os.makedirs("model_cache", exist_ok=True)
os.makedirs("violation_crops", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Inference device: {device}")

# ---------------------------------------------------------------------------
# SigLIP model + text features
# ---------------------------------------------------------------------------

logger.info(f"Loading SigLIP {SIGLIP_MODEL_NAME} ({SIGLIP_PRETRAINED})...")
siglip_model, _, siglip_preprocess = _run_quiet_startup(
    "SigLIP model init",
    lambda: open_clip.create_model_and_transforms(
        SIGLIP_MODEL_NAME, pretrained=SIGLIP_PRETRAINED,
    ),
)
siglip_model.eval().to(device) 

# Pre-compute text embeddings (once at startup)
tokenizer = _run_quiet_startup(
    "SigLIP tokenizer init",
    lambda: open_clip.get_tokenizer(SIGLIP_MODEL_NAME),
)
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
    warmup_size = getattr(getattr(siglip_model, "visual", None), "image_size", 224)
    if isinstance(warmup_size, (tuple, list)):
        warmup_h = int(warmup_size[0])
        warmup_w = int(warmup_size[1] if len(warmup_size) > 1 else warmup_size[0])
    else:
        warmup_h = warmup_w = int(warmup_size)
    siglip_model.encode_image(torch.zeros(1, 3, warmup_h, warmup_w).to(device))
logger.info("SigLIP model ready.")

siglip_logit_scale = float(siglip_model.logit_scale.exp().item())
_siglip_image_size = warmup_h


def _export_siglip_onnx_on_startup(model: torch.nn.Module, image_size: int, output_path: str) -> bool:
    """Export SigLIP image encoder to ONNX FP16 at startup. Returns True on success."""
    try:
        import copy

        class _EncoderFP16(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.visual = copy.deepcopy(m.visual).half()

            def forward(self, x):
                return self.visual(x.half()).float()

        encoder = _EncoderFP16(model).eval().cpu()
        dummy = torch.zeros(1, 3, image_size, image_size)

        logger.info("SigLIP ONNX FP16 export starting (this may take a minute)...")
        torch.onnx.export(
            encoder,
            dummy,
            output_path,
            dynamo=False,
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_features": {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        logger.info(f"SigLIP ONNX FP16 export complete -> {output_path}")
        return True
    except Exception as ex:
        logger.error(f"SigLIP ONNX export failed: {ex}")
        return False


def _load_siglip_onnx() -> object:
    """Load SigLIP ONNX session. Returns ORT InferenceSession or None."""
    if ort is None:
        logger.warning("onnxruntime not installed; SigLIP ONNX disabled")
        return None

    if not os.path.exists(SIGLIP_ONNX_PATH):
        if SIGLIP_ONNX_HF_REPO:
            try:
                from huggingface_hub import hf_hub_download
                logger.info("Downloading SigLIP ONNX from HF Hub (%s)...", SIGLIP_ONNX_HF_REPO)
                hf_hub_download(
                    repo_id=SIGLIP_ONNX_HF_REPO,
                    filename="siglip_image_encoder.onnx",
                    local_dir=os.path.dirname(SIGLIP_ONNX_PATH) or "model_cache",
                    token=os.getenv("HF_TOKEN"),
                )
                logger.info("SigLIP ONNX downloaded -> %s", SIGLIP_ONNX_PATH)
            except Exception as ex:
                logger.warning("HF Hub download failed: %s", ex)

        if not os.path.exists(SIGLIP_ONNX_PATH):
            if SIGLIP_ONNX_EXPORT_ON_STARTUP:
                ok = _export_siglip_onnx_on_startup(siglip_model, _siglip_image_size, SIGLIP_ONNX_PATH)
                if not ok:
                    return None
            else:
                logger.warning(
                    "SigLIP ONNX not found at %s. "
                    "Run: python scripts/export_siglip_onnx.py",
                    SIGLIP_ONNX_PATH,
                )
                return None

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = int(os.getenv("ORT_INTER_THREADS", "0"))
        sess_opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", "0"))
        session = ort.InferenceSession(
            SIGLIP_ONNX_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            sess_options=sess_opts,
        )
        active_provider = session.get_providers()[0]
        logger.info(f"SigLIP ONNX loaded from {SIGLIP_ONNX_PATH} (provider: {active_provider})")
        return session
    except Exception as ex:
        logger.warning(f"SigLIP ONNX load failed: {ex}; falling back to PyTorch")
        return None

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


def _load_effnet_bundle(run_id: str = EFFNET_RUN_ID):
    """
    Load EfficientNet stage-2 classifier with smart strategy.

    run_id: auto-discovered at startup via find_latest_effnet_run_id() — most recent
            finished run with export_onnx=success. Falls back to EFFNET_STABLE_RUN_ID.

    If PREFER_RECENT_PT=False (default):
      1. Stable ONNX from EFFNET_STABLE_RUN_ID
      2. Latest ONNX from auto-discovered run_id
      3. Local ONNX cache
      4. Latest PT from run_id
      5. PT fallback from disk

    If PREFER_RECENT_PT=True:
      1. Latest PT from auto-discovered run_id
      2. Stable ONNX
      3. Local ONNX cache
      4. PT fallback from disk
    """
    if not EFFNET_USE_ONNX:
        bundle = _load_effnet_pytorch()
        if bundle is not None:
            logger.info("EffNet PyTorch loaded (EFFNET_USE_ONNX=0, skipping remote fetch)")
            return bundle

    if EFFNET_PREFER_RECENT_PT:
        # Strategy 1: Prefer recent PT retrain (new models used immediately)
        logger.debug(f"Loading EffNet: Priority = RECENT_PT > STABLE_ONNX > LOCAL > FALLBACK_PT")

        # Try recent PT from retrain
        bundle = _load_effnet_pytorch_from_run(run_id)
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
        
        # Try latest ONNX from auto-discovered run (newer than stable)
        if run_id != EFFNET_STABLE_RUN_ID:
            bundle = _load_effnet_onnx_from_run(run_id)
            if bundle is not None:
                logger.info("✓ Loaded latest ONNX from auto-discovered run")
                return bundle

        # Try recent PT (newer, but untested)
        bundle = _load_effnet_pytorch_from_run(run_id)
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
    if ort is None:
        logger.warning("onnxruntime is not installed; skipping ONNX local load")
        return None

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
    if ort is None:
        logger.warning("onnxruntime is not installed; skipping ONNX MLflow load")
        return None

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



_effnet_run_id = find_latest_effnet_run_id(EFFNET_EXPERIMENT_NAME, fallback=EFFNET_RUN_ID)

_siglip_ort_session = _load_siglip_onnx() if SIGLIP_ONNX_ENABLED else None

# Bundle for classifier
_siglip_bundle: dict
if _siglip_ort_session is not None:
    _siglip_bundle = {
        "model": _siglip_ort_session,
        "preprocess": siglip_preprocess,
        "text_features": siglip_text_features,
        "format": "onnx",
        "logit_scale": siglip_logit_scale,
    }
    # PyTorch model no longer needed — free ~1.6 GB RAM.
    del siglip_model
    torch.cuda.empty_cache()
    logger.info("SigLIP inference: ONNX FP16 (PyTorch model offloaded)")
else:
    _siglip_bundle = {
        "model": siglip_model,
        "preprocess": siglip_preprocess,
        "text_features": siglip_text_features,
        "logit_scale": siglip_logit_scale,
    }

model_stage2 = {
    "siglip": _siglip_bundle,
    "effnet": _load_effnet_bundle(run_id=_effnet_run_id),
}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=INFERENCE_THREAD_WORKERS)

app = FastAPI(title="SentinelVision API", version="3.0.0")
app.mount(
    "/violation_crops",
    StaticFiles(directory="violation_crops"),
    name="violation_crops",
)
templates = Jinja2Templates(directory="templates")