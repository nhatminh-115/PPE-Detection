import os
from dotenv import load_dotenv
load_dotenv()

USE_EMA              = True   
BOX_EMA_ALPHA        = 0.6    
EMA_ALPHA            = 0.3
HYSTERESIS_MARGIN    = 0.125
ALERT_TIME_THRESHOLD = 3.0
GRACE_PERIOD_SEC     = 10.0

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI")

CONF_M1       = 0.05
CONF_M2       = 0.05
IOU_NMS       = 0.60
WBF_IOU       = 0.35
SKIP_THR      = 0.08
MODEL_WEIGHTS = [2, 1]

LABEL_COLS    = ['hardhat', 'vest']
NUM_CLASSES   = 4
IMG_SIZE      = 224
ROUTER_MIN_SIDE_PX = float(os.getenv("ROUTER_MIN_SIDE_PX", "90"))
CLIP_USE_POSE = os.getenv("CLIP_USE_POSE", "1").strip().lower() not in {"0", "false", "no"}

# CLIP zero-shot config
CLIP_MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED = 'laion2b_s34b_b79k'
CLIP_PROMPTS = {
    'hardhat': {
        'positive': [
            'a construction worker wearing a hard hat',
            'a person wearing a safety helmet on their head',
            'a worker with a hardhat',
            'a worker wearing a white safety helmet',
            'a person wearing a white hard hat on a construction site',
        ],
        'negative': [
            'a person without a hard hat on a construction site',
            'a person with bare head and no helmet',
            'a worker without head protection',
            'a person wearing a white cap instead of a hard hat',
        ],
    },
    'vest': {
        'positive': [
            'a person wearing a high-visibility safety vest',
            'a worker wearing a reflective safety vest',
            'a person in a fluorescent safety vest',
        ],
        'negative': [
            'a person without a safety vest',
            'a person in regular clothing without a vest',
            'a worker without a reflective vest',
        ],
    },
}

# Inference strategy — detect → lock → cooldown
CLASSIFY_ACTIVE_SEC   = 12.0   # classify for first 12 seconds 
CLASSIFY_COOLDOWN_SEC = float(os.getenv("CLASSIFY_COOLDOWN_SEC", "45"))
CLASSIFY_FORCE_RECHECK_SEC = float(os.getenv("CLASSIFY_FORCE_RECHECK_SEC", "20"))
CLASSIFY_UNCERTAIN_MARGIN = float(os.getenv("CLASSIFY_UNCERTAIN_MARGIN", "0.08"))
CLIP_LOCK_MIN_CONF = float(os.getenv("CLIP_LOCK_MIN_CONF", "0.78"))
HARDHAT_VIS_MIN_SIDE = int(os.getenv("HARDHAT_VIS_MIN_SIDE", "44"))
HARDHAT_VIS_MIN_BRIGHT = float(os.getenv("HARDHAT_VIS_MIN_BRIGHT", "36"))
HARDHAT_VIS_MIN_SHARP = float(os.getenv("HARDHAT_VIS_MIN_SHARP", "28"))
WHITE_HARDHAT_PRIOR_ENABLE = os.getenv("WHITE_HARDHAT_PRIOR_ENABLE", "1").strip().lower() not in {"0", "false", "no"}
WHITE_HARDHAT_SAT_MAX = float(os.getenv("WHITE_HARDHAT_SAT_MAX", "58"))
WHITE_HARDHAT_VAL_MIN = float(os.getenv("WHITE_HARDHAT_VAL_MIN", "148"))
WHITE_HARDHAT_MIN_RATIO = float(os.getenv("WHITE_HARDHAT_MIN_RATIO", "0.07"))
WHITE_HARDHAT_PROB_BOOST = float(os.getenv("WHITE_HARDHAT_PROB_BOOST", "0.12"))

PPE_THRESHOLDS = {
    'hardhat': {'ok': 0.70, 'warn': 0.40}, 
    'vest':    {'ok': 0.50, 'warn': 0.15}  
}

STATE_COLORS = {
    2: (0, 200, 0),
    1: (0, 255, 255),
    0: (0, 50, 255)
}

DETECT_EVERY  = 3    
COLOR_TRACK   = (0, 200, 255)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI")