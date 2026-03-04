import os
import cv2
import time
import shutil
import torch
import logging
import numpy as np
import timm
import concurrent.futures
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import supervision as sv

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False

import mlflow
from supabase import create_client

# ==============================================================================
# 1. CONFIGURATION & LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

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
    'vest':    {'ok': 0.50, 'warn': 0.20}  
}

STATE_COLORS = {
    2: (0, 200, 0),
    1: (0, 255, 255),
    0: (0, 50, 255)
}

DETECT_EVERY  = 3    
COLOR_TRACK   = (0, 200, 255)

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

os.makedirs("model_cache", exist_ok=True)
os.makedirs("violation_crops", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

# ==============================================================================
# 2. EXTERNAL SERVICES (LAZY INITIALIZATION)
# ==============================================================================
supabase_client = None
db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
reported_ids = set()

def get_supabase_client():
    """Implements lazy initialization for Supabase to prevent CI/CD import crashes."""
    global supabase_client
    if supabase_client is None:
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Supabase connection successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
        else:
            logger.warning("Supabase credentials missing. Remote logging disabled.")
    return supabase_client

def log_violation_to_supabase(tracker_id, missing_items, missing_probs, crop_img):
    """Asynchronously dispatches violation events and localized crops to Supabase."""
    try:
        timestamp_now = int(time.time())
        img_filename = f"violation_crops/ID{tracker_id}_{timestamp_now}.jpg"
        
        if crop_img is not None and crop_img.size > 0:
            cv2.imwrite(img_filename, crop_img)
            
        violations_list = [
            {
                "tracker_id": int(tracker_id),
                "image_path": img_filename,
                "violation_type": f"none_{item}",
                "confidence": float(prob)
            }
            for item, prob in zip(missing_items, missing_probs)
        ]
            
        data = {
            "violations": violations_list,
            "status": "Warning"
        }
        
        client = get_supabase_client()
        if client:
            client.table("ppe_violations").insert(data).execute()
            logger.info(f"Database sync successful for violation ID: {tracker_id}.")
    except Exception as e:
        logger.error(f"Database logging execution failed: {e}")

def pull_artifact_from_mlflow_run(run_id, artifact_path, cache_dir="model_cache"):
    """Fetches model artifacts idempotently from MLflow registry."""
    if not MLFLOW_URI:
        logger.warning("MLFLOW_TRACKING_URI is not set. Assuming local artifact execution.")
        return os.path.join(cache_dir, f"{run_id}_{os.path.basename(artifact_path)}")
        
    file_name = os.path.basename(artifact_path)
    cached_file_path = os.path.join(cache_dir, f"{run_id}_{file_name}")
    
    if os.path.exists(cached_file_path):
        logger.info(f"Cache HIT. Utilizing local artifact: {cached_file_path}")
        return cached_file_path
        
    mlflow.set_tracking_uri(MLFLOW_URI)
    logger.info(f"Cache MISS. Fetching artifact from remote Run ID: {run_id[:8]}...")
    
    model_uri = f"runs:/{run_id}/{artifact_path}"
    downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=cache_dir)
    os.rename(downloaded_path, cached_file_path)
    
    logger.info(f"Artifact fetched and cached successfully.")
    return cached_file_path

# ==============================================================================
# 3. GLOBAL INFERENCE ENGINES
# ==============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Inference Device initialized: {device} | WBF Engine: {WBF_AVAILABLE}")

model1_path = pull_artifact_from_mlflow_run(run_id="b52f0641a3fe41899bb4c620fdef053d", artifact_path="weights/best.pt")
model_stage2_path = pull_artifact_from_mlflow_run(run_id="e44391ef94b54ce3b867d46b7ce33ec3", artifact_path="weights/best_stage2_effnet.pt")
model2_path = "yolo26l.pt"

# Load Stage 1 Ensembles
model1 = YOLO(model1_path)
model1.predict(torch.zeros((1, 3, 1280, 1280)).to(device), imgsz=1280) 
model2 = YOLO(model2_path)

# Load Stage 2 Classifier
model_stage2 = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=NUM_CLASSES)
model_stage2.load_state_dict(torch.load(model_stage2_path, map_location=device))
model_stage2.eval().to(device)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# FastAPI Initialization
app = FastAPI(title="PPE Detection OS API", version="3.0.0")
app.mount("/violation_crops", StaticFiles(directory="violation_crops"), name="violation_crops")
templates = Jinja2Templates(directory="templates")

class StreamState:
    def __init__(self):
        self.source = None
        self.trigger_restart = False

stream_state = StreamState()

# ==============================================================================
# 4. CORE ALGORITHMIC LOGIC
# ==============================================================================
def extract_boxes(results, img_w, img_h, class_filter=None):
    boxes, scores, labels = [], [], []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if class_filter is not None and cls not in class_filter: continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
            scores.append(float(box.conf[0]))
            labels.append(cls)
    return boxes, scores, labels

def fuse_boxes(b1, s1, l1, b2, s2, l2, img_w, img_h):
    if not (b1 or b2): return np.empty((0, 4)), np.array([])
    boxes_list  = [b1, b2] if (b1 and b2) else ([b1] if b1 else [b2])
    scores_list = [s1, s2] if (b1 and b2) else ([s1] if b1 else [s2])
    labels_list = [[float(x) for x in l1], [float(x) for x in l2]] if (b1 and b2) else [[float(x) for x in (l1 if b1 else l2)]]
    w = MODEL_WEIGHTS if (b1 and b2) else [1]

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=w, iou_thr=WBF_IOU, skip_box_thr=SKIP_THR,
    )
    if len(fused_boxes) == 0: return np.empty((0, 4)), np.array([])
    
    fused_boxes = np.array(fused_boxes)
    fused_boxes[:, [0, 2]] *= img_w
    fused_boxes[:, [1, 3]] *= img_h
    return fused_boxes, np.array(fused_scores, dtype=float)

def classify_ppe_batch(model_stage2, frame, boxes, device):
    """Executes EfficientNetV2 inference and XAI Forward-CAM Color Penalization."""
    img_h, img_w = frame.shape[:2]
    tensors, valid_idx, raw_crops = [], [], []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        pw, ph = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
        x1c, y1c = max(0, x1 - pw), max(0, y1 - ph)
        x2c, y2c = min(img_w, x2 + pw), min(img_h, y2 + ph)

        crop_w, crop_h = x2c - x1c, y2c - y1c
        if crop_w < 10 or crop_h < 15: continue
        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0: continue

        if crop_w < 80 or crop_h < 100:
            scale = max(80 / crop_w, 100 / crop_h)
            crop  = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_LANCZOS4)

        raw_crops.append(crop) 
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(val_transform(pil_img))
        valid_idx.append(i)

    if not tensors: return [None] * len(boxes)

    batch = torch.stack(tensors).to(device)
    
    with torch.no_grad():
        features = model_stage2.forward_features(batch)
        pooled   = model_stage2.global_pool(features)
        logits   = model_stage2.classifier(pooled)
        probs    = torch.sigmoid(logits).cpu().numpy()

        # Generate Class Activation Map (CAM) for the hardhat neuron
        weight_hardhat = model_stage2.classifier.weight[0] 
        cam = torch.einsum('bchw,c->bhw', features, weight_hardhat) 
        cam = torch.relu(cam) 

    for idx_in_batch, original_idx in enumerate(valid_idx):
        prob_hardhat = probs[idx_in_batch][0]
        
        # XAI Intervention: Trigger Color Penalizer if probability is anomalously high
        if prob_hardhat > 0.4:
            c = cam[idx_in_batch]
            c_max = c.max()
            if c_max > 1e-4: 
                c = c / c_max
            
            c_np = c.cpu().numpy()
            crop_img = raw_crops[idx_in_batch]
            h, w = crop_img.shape[:2]
            c_resized = cv2.resize(c_np, (w, h))
            
            # Isolate highly activated spatial regions
            mask = c_resized > 0.5 
            
            if mask.sum() > 10: 
                hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                
                s_channel = hsv[:, :, 1]
                v_channel = hsv[:, :, 2] 
                
                mean_saturation = s_channel[mask].mean()
                mean_brightness = v_channel[mask].mean()
                
                # Dual-Penalty Heuristic execution
                if mean_brightness < 70:
                    probs[idx_in_batch][0] -= 0.55
                elif mean_saturation < 80:
                    probs[idx_in_batch][0] -= 0.35

                probs[idx_in_batch][0] = max(0.0, probs[idx_in_batch][0])

    results = [None] * len(boxes)
    for idx_in_batch, original_idx in enumerate(valid_idx):
        results[original_idx] = {"probs": probs[idx_in_batch]}
    return results

def get_next_state(class_name, prob, current_state):
    """Evaluates finite state machine transitions utilizing mathematical hysteresis."""
    th = PPE_THRESHOLDS[class_name]
    if current_state is None:
        if prob >= th['ok']:   return 2
        if prob >= th['warn']: return 1
        return 0
    if current_state == 2:
        if prob < th['ok'] - HYSTERESIS_MARGIN: return 1 if prob >= th['warn'] else 0
        return 2
    elif current_state == 1:
        if prob >= th['ok'] + HYSTERESIS_MARGIN: return 2
        if prob < th['warn'] - HYSTERESIS_MARGIN: return 0
        return 1
    else: 
        if prob >= th['warn'] + HYSTERESIS_MARGIN: return 2 if prob >= th['ok'] + HYSTERESIS_MARGIN else 1
        return 0

def update_ema_and_decide(raw_results, tracker_ids, boxes, frame, ppe_ema, ppe_state, violation_timer, current_time):
    """Processes temporal smoothing and orchestrates violation accumulation logic."""
    smoothed = []

    for tid, result, box in zip(tracker_ids, raw_results, boxes):
        if result is None:
            smoothed.append(None)
            continue

        curr_prob = result["probs"]
        if USE_EMA:
            if tid not in ppe_ema: ppe_ema[tid] = curr_prob.copy()
            else: ppe_ema[tid] = EMA_ALPHA * curr_prob + (1.0 - EMA_ALPHA) * ppe_ema[tid]
            avg_probs = ppe_ema[tid]
        else:
            avg_probs = curr_prob

        current_states = ppe_state.get(tid, [None] * len(LABEL_COLS))
        new_states = [get_next_state(LABEL_COLS[i], avg_probs[i], current_states[i]) for i in range(len(LABEL_COLS))]

        if tid not in violation_timer:
            violation_timer[tid] = {"duration": 0.0, "last_tick": current_time}

        elapsed = current_time - violation_timer[tid]["last_tick"]
        violation_timer[tid]["last_tick"] = current_time

        min_state = min(new_states)
        if min_state < 2:
            violation_timer[tid]["duration"] += elapsed
        else:
            violation_timer[tid]["duration"] = 0.0

        if (violation_timer[tid]["duration"] >= ALERT_TIME_THRESHOLD and tid not in reported_ids):
            reported_ids.add(tid) 

            missing_items = [LABEL_COLS[i] for i, s in enumerate(new_states) if s == 0]
            missing_probs = [avg_probs[i] for i, s in enumerate(new_states) if s == 0]

            if missing_items:
                img_h, img_w = frame.shape[:2]
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                crop = frame[y1:y2, x1:x2].copy()
                db_executor.submit(log_violation_to_supabase, tid, missing_items, missing_probs, crop)

        ppe_state[tid] = new_states
        smoothed.append({"probs": avg_probs, "states": new_states})

    return smoothed

def garbage_collection(active_ids, current_time, last_seen_timer, *state_dicts):
    """Executes state cleanup utilizing Time-To-Live (TTL) grace periods."""
    for tid in active_ids:
        last_seen_timer[tid] = current_time
        
    stale_keys = [k for k, t in last_seen_timer.items() if current_time - t > GRACE_PERIOD_SEC]
    
    for k in stale_keys:
        for d in state_dicts:
            if k in d:
                del d[k]
        reported_ids.discard(k)
        del last_seen_timer[k]

def draw_ppe_result(frame, box, ppe_result, tracker_id):
    """Renders real-time bounding boxes and probabilistic telemetry on the active frame."""
    x1, y1, x2, y2 = map(int, box)
    if ppe_result is None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TRACK, 2)
        cv2.putText(frame, f"ID:{tracker_id}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TRACK, 1)
        return

    states = ppe_result["states"]
    min_state = min(states)
    main_color = STATE_COLORS[min_state]

    label_parts = []
    for i, state in enumerate(states):
        name = LABEL_COLS[i][:4].upper()
        if state == 0:   label_parts.append(f"MISS {name}")
        elif state == 1: label_parts.append(f"WARN {name}")
    
    label_text = f"ID:{tracker_id} " + ("- ".join(label_parts) if label_parts else "ALL OK")

    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), main_color, -1)
    text_color = (0, 0, 0) if min_state == 1 else (255, 255, 255)
    cv2.putText(frame, label_text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    bar_y = y2 + 4
    for i, (lbl, prob, state) in enumerate(zip(LABEL_COLS, ppe_result["probs"], states)):
        bar_color = STATE_COLORS[state]
        cv2.rectangle(frame, (x1, bar_y + i*11), (x1 + int(prob * 50), bar_y + i*11 + 7), bar_color, -1)
        cv2.putText(frame, f"{lbl[:4]}:{prob:.2f}", (x1 + 53, bar_y + i*11 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

# ==============================================================================
# 5. PIPELINE GENERATOR & API ROUTES
# ==============================================================================
def generate_frames():
    """Asynchronous frame generator maintaining core pipeline state."""
    tracker = sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=240,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=2,
    )

    last_tracked     = sv.Detections.empty()
    last_ppe_results = []
    ppe_ema, box_ema, ppe_state, violation_timer, last_seen_timer = {}, {}, {}, {}, {}
    
    frame_count = 0
    cap = None

    while True:
        if cap is None or stream_state.trigger_restart:
            if cap is not None:
                cap.release()
            
            if not stream_state.source:
                time.sleep(1)
                continue
                
            logger.info(f"Establishing stream connection: {stream_state.source}")
            cap = cv2.VideoCapture(stream_state.source)
            stream_state.trigger_restart = False
            
            if not cap.isOpened():
                logger.error("Failed to acquire video stream.")
                cap = None
                stream_state.source = None
                continue

        ret, frame = cap.read()
        if not ret: 
            logger.info("Video stream reached EOF.")
            cap.release()
            cap = None
            stream_state.source = None
            continue

        frame_count += 1
        img_h, img_w = frame.shape[:2]
        current_time = time.time()

        if frame_count % DETECT_EVERY == 0:
            future1 = executor.submit(model1.predict, frame, conf=CONF_M1, iou=IOU_NMS, device=device, imgsz=1280, verbose=False)
            future2 = executor.submit(model2.predict, frame, conf=CONF_M2, iou=IOU_NMS, device=device, verbose=False, classes=[0], imgsz=1280)
            
            res1, res2 = future1.result(), future2.result()
            b1, s1, l1 = extract_boxes(res1, img_w, img_h)
            b2, s2, l2 = extract_boxes(res2, img_w, img_h, class_filter=[0])

            fused_boxes, fused_scores = fuse_boxes(b1, s1, l1, b2, s2, l2, img_w, img_h) if WBF_AVAILABLE else (np.array(b1) * np.array([img_w, img_h, img_w, img_h]) if b1 else np.empty((0, 4)), np.array(s1, dtype=float) if b1 else np.array([]))

            if len(fused_boxes) > 0:
                tracked = tracker.update_with_detections(sv.Detections(xyxy=fused_boxes.astype(float), confidence=fused_scores, class_id=np.zeros(len(fused_boxes), dtype=int)))
            else:
                tracked = sv.Detections.empty()

            if len(tracked) > 0 and tracked.tracker_id is not None:
                smoothed_boxes = []
                for tid, box in zip(tracked.tracker_id, tracked.xyxy):
                    if tid not in box_ema: box_ema[tid] = box
                    else: box_ema[tid] = BOX_EMA_ALPHA * box + (1.0 - BOX_EMA_ALPHA) * box_ema[tid]
                    smoothed_boxes.append(box_ema[tid])
                tracked.xyxy = np.array(smoothed_boxes)

                raw_results = classify_ppe_batch(model_stage2, frame, tracked.xyxy, device)
                ppe_results = update_ema_and_decide(raw_results, tracked.tracker_id, tracked.xyxy, frame, ppe_ema, ppe_state, violation_timer, current_time)
                garbage_collection(tracked.tracker_id, current_time, last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer)
            else:
                ppe_results = []
                garbage_collection([], current_time, last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer)

            last_tracked, last_ppe_results = tracked, ppe_results
        else:
            tracked, ppe_results = last_tracked, last_ppe_results

        annotated = frame.copy()
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for box, tid, ppe_result in zip(tracked.xyxy, tracked.tracker_id, ppe_results):
                draw_ppe_result(annotated, box, ppe_result, tid)
        
        target_height = 720
        scale = target_height / img_h
        target_width = int(img_w * scale)
        display = cv2.resize(annotated, (target_width, target_height))
        
        ret, buffer = cv2.imencode('.jpg', display)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Entrypoint for the Single Page Application UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    """Endpoint for streaming MJPEG inference data."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload_video")
async def upload_video(video_file: UploadFile = File(...)):
    """Handles dynamic video injections for the pipeline."""
    try:
        file_path = f"temp_uploads/{video_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        stream_state.source = file_path
        stream_state.trigger_restart = True
        
        return {"status": "success", "message": f"Source injected: {video_file.filename}"}
    except Exception as e:
        logger.error(f"Video injection failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/violations")
def get_violations():
    """Retrieves recent violation payloads from the Supabase registry."""
    client = get_supabase_client()
    if not client: return {"data": []}
    res = client.table("ppe_violations").select("*").order("created_at", desc=True).limit(20).execute()
    return {"data": res.data}

@app.post("/api/flush")
def flush_images():
    """Purges the local blob storage for evidence crops."""
    try:
        count = 0
        for f in os.listdir("violation_crops"):
            file_path = os.path.join("violation_crops", f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        logger.info(f"Local storage purged. {count} objects removed.")
        return {"status": "success", "message": f"Flushed {count} objects."}
    except Exception as e:
        logger.error(f"Storage purge failed: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)