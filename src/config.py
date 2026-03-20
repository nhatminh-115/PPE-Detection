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