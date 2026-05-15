import os
from dotenv import load_dotenv

_env = os.getenv("APP_ENV", "prod")
_env_file = f".env.{_env}"
if os.path.exists(_env_file):
    load_dotenv(_env_file, override=True)
else:
    load_dotenv()

# ---------------------------------------------------------------------------
# External services  (loaded from .env)
# ---------------------------------------------------------------------------
SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_KEY          = os.getenv("SUPABASE_KEY")           # anon key — read-only dashboard use
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")   # service_role key — backend writes
MLFLOW_URI            = os.getenv("MLFLOW_TRACKING_URI")

# ---------------------------------------------------------------------------
# Model paths / identifiers
# ---------------------------------------------------------------------------
MLFLOW_RUN_ID        = "b52f0641a3fe41899bb4c620fdef053d"
MLFLOW_ARTIFACT_PATH = "weights/best.pt"
MODEL2_PATH          = "yolo26l.pt"
MODEL_POSE_PATH      = "yolo26m-pose.pt"

# EfficientNetV2-B0 Classifier
# EFFNET_STABLE_RUN_ID: Verified, production-stable run with ONNX export (always available for fallback)
# EFFNET_RUN_ID: Latest good retrain run (may be updated after each drift trigger)
EFFNET_STABLE_RUN_ID  = "af05de89dacb4aaf893c68a2e4552ba3"  # Stable, ✓ ONNX available
EFFNET_RUN_ID         = os.getenv("EFFNET_RUN_ID", "af05de89dacb4aaf893c68a2e4552ba3")  # overridden at startup by MLflow discovery
EFFNET_EXPERIMENT_NAME = "PPE_Stage2_Classification_"

# ONNX format (artifact stored in MLflow run)
EFFNET_ONNX_ARTIFACT_PATH    = "weights/effnet_ppe.onnx"
EFFNET_ONNX_LOCAL_FALLBACK   = "model_cache/effnet_ppe_dynamic.onnx"
EFFNET_ONNX_INT8_FALLBACK    = "model_cache/effnet_ppe_int8.onnx"

# PyTorch format (retrain result, newer than ONNX)
EFFNET_PT_ARTIFACT_PATH      = "weights/best_stage2_effnet.pt"
EFFNET_PT_LOCAL_FALLBACK     = "model_cache/e44391ef94b54ce3b867d46b7ce33ec3_best_stage2_effnet.pt"

# Loading strategy
EFFNET_USE_ONNX              = os.getenv("EFFNET_USE_ONNX", "1") == "1"
EFFNET_PREFER_RECENT_PT      = False  # Prefer recent PT retrain over stable ONNX

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
CONF_M1       = 0.05
CONF_M2       = 0.05
IOU_NMS       = 0.60
WBF_IOU       = 0.35
SKIP_THR      = 0.08
MODEL_WEIGHTS = [2, 1]
DETECT_EVERY  = 3

# ---------------------------------------------------------------------------
# EffNet / classifier image size
# ---------------------------------------------------------------------------
LABEL_COLS  = ['hardhat', 'vest']
NUM_CLASSES = 4
IMG_SIZE    = 224

# ---------------------------------------------------------------------------
# Tracking & state machine
# ---------------------------------------------------------------------------
USE_EMA              = True
BOX_EMA_ALPHA        = 0.6
EMA_ALPHA            = 0.5
HYSTERESIS_MARGIN    = 0.125
ALERT_TIME_THRESHOLD = 3.0
GRACE_PERIOD_SEC     = 10.0

# ---------------------------------------------------------------------------
# PPE decision thresholds
# ---------------------------------------------------------------------------
PPE_THRESHOLDS = {
    'hardhat': {'ok': 0.60, 'warn': 0.40},
    'vest':    {'ok': 0.50, 'warn': 0.25},
}

# ---------------------------------------------------------------------------
# Classification strategy  (detect → lock → cooldown)
# ---------------------------------------------------------------------------
ROUTER_MIN_SIDE_PX           = 110
SIGLIP_USE_POSE              = True
CLASSIFY_ACTIVE_SEC          = 20.0
CLASSIFY_FORCE_RECHECK_SEC   = 20.0
CLASSIFY_UNCERTAIN_MARGIN    = 0.08
SIGLIP_LOCK_MIN_CONF         = 0.82
CLASSIFY_LOCK_MIN_CONSISTENT = 3
VIOLATION_COOLDOWN_SEC       = 300.0  # 5 minutes before re-reporting same tracker

# ---------------------------------------------------------------------------
# Hardhat visibility gate
# ---------------------------------------------------------------------------
HARDHAT_VIS_MIN_SIDE   = 44
HARDHAT_VIS_MIN_BRIGHT = 36.0
HARDHAT_VIS_MIN_SHARP  = 28.0

# ---------------------------------------------------------------------------
# Color-based priors (HSV)
# ---------------------------------------------------------------------------
WHITE_HARDHAT_PRIOR_ENABLE = True
WHITE_HARDHAT_SAT_MAX      = 58.0
WHITE_HARDHAT_VAL_MIN      = 148.0
WHITE_HARDHAT_MIN_RATIO    = 0.07
WHITE_HARDHAT_PROB_BOOST   = 0.12

BRIGHT_HARDHAT_PRIOR_ENABLE = True
BRIGHT_HARDHAT_MIN_RATIO    = 0.08
BRIGHT_HARDHAT_PROB_BOOST   = 0.10

YELLOW_HARDHAT_H_MIN = 15.0
YELLOW_HARDHAT_H_MAX = 40.0
YELLOW_HARDHAT_S_MIN = 55.0
YELLOW_HARDHAT_V_MIN = 120.0

DARK_HEAD_PRIOR_ENABLE = True
DARK_HEAD_VAL_MAX      = 72.0
DARK_HEAD_MIN_RATIO    = 0.22
DARK_HEAD_PROB_PENALTY = 0.14

# ---------------------------------------------------------------------------
# SigLIP zero-shot model + prompts
# ---------------------------------------------------------------------------
SIGLIP_MODEL_NAME = 'ViT-SO400M-14-SigLIP'
SIGLIP_PRETRAINED = 'webli'
SIGLIP_ONNX_ENABLED = os.getenv("SIGLIP_ONNX_ENABLED", "0") == "1"
SIGLIP_ONNX_EXPORT_ON_STARTUP = os.getenv("SIGLIP_ONNX_EXPORT_ON_STARTUP", "0") == "1"
SIGLIP_ONNX_PRECISION = os.getenv("SIGLIP_ONNX_PRECISION", "fp16").strip().lower()
SIGLIP_ONNX_PATH = os.getenv("SIGLIP_ONNX_PATH", "model_cache/siglip_image_encoder.onnx")
SIGLIP_ONNX_HF_REPO = os.getenv("SIGLIP_ONNX_HF_REPO", "Nhatminh1234/siglip-so400m-ppe-fp16")
SIGLIP_PROMPTS = {
    'hardhat': {
        'positive': [
            'a construction worker wearing a hard hat',
            'a person wearing a safety helmet on their head',
            'a worker with a hardhat',
            'a worker wearing a bright safety helmet',
            'a person wearing a bright hard hat on a construction site',
            'a person wearing a white hard hat',
            'a person wearing a yellow hard hat',
        ],
        'negative': [
            'a person without a hard hat',
            'a person with bare head and no helmet',
            'a worker without head protection',
            'a person wearing dark headwear',
            'a person with a shadowed head and no hard hat',
            'a person with black hair and no hard hat',
        ],
    },
    'vest': {
        'positive': [
            'a person wearing a high-visibility safety vest',
            'a worker wearing a reflective safety vest',
            'a person in a fluorescent safety vest',
            'a construction worker wearing a reflective yellow safety vest with reflective strips',
            'a person wearing PPE safety vest on a construction site',
            'a worker wearing an orange high-visibility vest',
            'a worker wearing a lime green reflective vest',
            'a person wearing a safety vest even if partially occluded',
            'a person wearing a reflective vest in low light',
            'a distant construction worker wearing a safety vest',
            'top down view of a person wearing a safety vest',
            'aerial view of a person wearing a safety vest',
            'bird eye view of a person wearing a safety vest',
            'a worker wearing an open safety vest without fastening',
            'construction worker with unfastened high-vis vest draped over shoulders',
            'person wearing a loose reflective vest open at the front',
            'a person wearing a dusty or dirty reflective safety vest',
            'a person with reflective strips visible on a safety vest',
            'a worker wearing a high-vis vest over dark clothes',
        ],
        'negative': [
            'a person without a safety vest',
            'a person in regular clothing without a vest',
            'a worker without a reflective vest',
            'a person in a t-shirt with no safety vest',
            'a person in a shirt or jacket with no high-visibility vest',
            'a worker with bare torso area and no vest',
            'aerial view of a person in regular clothing without a safety vest',
            'top down view of a person without a safety vest',
            'construction worker without any reflective strips on torso clothing',
        ],
    },
}

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
STATE_COLORS = {
    2: (0, 200, 0),
    1: (0, 255, 255),
    0: (0, 50, 255),
}
COLOR_TRACK = (0, 200, 255)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
INFERENCE_THREAD_WORKERS = 3
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
