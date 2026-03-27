import os
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# External services  (loaded from .env)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI")

# ---------------------------------------------------------------------------
# Model paths / identifiers
# ---------------------------------------------------------------------------
MLFLOW_RUN_ID        = "b52f0641a3fe41899bb4c620fdef053d"
MLFLOW_ARTIFACT_PATH = "weights/best.pt"
MODEL2_PATH          = "yolo26l.pt"
MODEL_POSE_PATH      = "yolo26m-pose.pt"
EFFNET_VARIANT       = "auto"   # auto | base | attention
EFFNET_WEIGHTS_PATH  = ""       # override explicit path; empty = auto-discover

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
EMA_ALPHA            = 0.3
HYSTERESIS_MARGIN    = 0.125
ALERT_TIME_THRESHOLD = 3.0
GRACE_PERIOD_SEC     = 10.0

# ---------------------------------------------------------------------------
# PPE decision thresholds
# ---------------------------------------------------------------------------
PPE_THRESHOLDS = {
    'hardhat': {'ok': 0.70, 'warn': 0.40},
    'vest':    {'ok': 0.50, 'warn': 0.15},
}

# ---------------------------------------------------------------------------
# Classification strategy  (detect → lock → cooldown)
# ---------------------------------------------------------------------------
ROUTER_MIN_SIDE_PX           = 110
CLIP_USE_POSE                = True
CLASSIFY_ACTIVE_SEC          = 20.0
CLASSIFY_COOLDOWN_SEC        = 45.0
CLASSIFY_FORCE_RECHECK_SEC   = 20.0
CLASSIFY_UNCERTAIN_MARGIN    = 0.08
CLIP_LOCK_MIN_CONF           = 0.82
CLASSIFY_LOCK_MIN_CONSISTENT = 3

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
# CLIP zero-shot model + prompts
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = 'ViT-SO400M-14-SigLIP'
CLIP_PRETRAINED = 'webli'
CLIP_PROMPTS = {
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
        ],
    },
    'vest': {
        'positive': [
            'a person wearing a high-visibility safety vest',
            'a worker wearing a reflective safety vest',
            'a person in a fluorescent safety vest',
            'a construction worker wearing a reflective yellow safety vest with reflective strips',
            'a person wearing PPE safety vest on a construction site',
            'a person wearing lifejacket',
            'a person wearing water rescue vest',
            'a person wearing bouyancy vest',
        ],
        'negative': [
            'a person without a safety vest',
            'a person in regular clothing without a vest',
            'a worker without a reflective vest',
            'a person wearing a life jacket instead of a construction safety vest',
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
