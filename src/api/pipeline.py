import time
import logging
import queue
import threading
import numpy as np
import supervision as sv
from src.api.app import app, model1, model2, model_pose, model_stage2, device
from src.config import (
    CONF_M1, CONF_M2, IOU_NMS, DETECT_EVERY, BOX_EMA_ALPHA,
    CLASSIFY_ACTIVE_SEC,
    CLASSIFY_FORCE_RECHECK_SEC, CLASSIFY_UNCERTAIN_MARGIN,
    SIGLIP_LOCK_MIN_CONF, PPE_THRESHOLDS,
    CLASSIFY_LOCK_MIN_CONSISTENT,
)
from src.inference import (
    extract_boxes, fuse_boxes, WBF_AVAILABLE,
    classify_ppe_batch, should_run_pose_for_boxes, update_ema_and_decide,
    garbage_collection, StreamState,
)
import cv2
import uuid

logger = logging.getLogger(__name__)

_FRAME_QUEUE_SIZE  = 3
_RESULT_QUEUE_SIZE = 3
_STOP              = object()
POSE_EVERY         = 9

# Latest encoded frame per camera — for snapshot endpoint
_latest_frames: dict[str, bytes] = {}


def _is_verdict_stable(verdict: dict | None) -> bool:
    if not verdict or "probs" not in verdict:
        return False

    probs = verdict["probs"]
    if probs is None or len(probs) < 2:
        return False

    model_name = str(verdict.get("router_model", "")).lower()
    min_certainty = float(SIGLIP_LOCK_MIN_CONF) if model_name in ("siglip", "clip") else 0.72

    for i, label in enumerate(("hardhat", "vest")):
        p = float(probs[i])
        certainty = max(p, 1.0 - p)
        if certainty < min_certainty:
            return False

        th = PPE_THRESHOLDS[label]
        dist_warn = abs(p - float(th["warn"]))
        dist_ok = abs(p - float(th["ok"]))
        if min(dist_warn, dist_ok) < float(CLASSIFY_UNCERTAIN_MARGIN):
            return False

    return True


def _verdict_signature(verdict: dict | None):
    if not verdict or "probs" not in verdict:
        return None
    probs = verdict["probs"]
    if probs is None or len(probs) < 2:
        return None

    states = []
    for i, label in enumerate(("hardhat", "vest")):
        p = float(probs[i])
        th = PPE_THRESHOLDS[label]
        if p >= float(th["ok"]):
            states.append(2)
        elif p >= float(th["warn"]):
            states.append(1)
        else:
            states.append(0)

    return (str(verdict.get("router_model", "")), states[0], states[1])


def _update_lock_consensus(lock: dict, verdict: dict | None):
    if not _is_verdict_stable(verdict):
        lock["candidate_sig"] = None
        lock["candidate_count"] = 0
        return

    sig = _verdict_signature(verdict)
    if sig is None:
        lock["candidate_sig"] = None
        lock["candidate_count"] = 0
        return

    if lock.get("candidate_sig") == sig:
        lock["candidate_count"] = int(lock.get("candidate_count", 0)) + 1
    else:
        lock["candidate_sig"] = sig
        lock["candidate_count"] = 1


# ---------------------------------------------------------------------------
# Zone filtering
# ---------------------------------------------------------------------------

def _filter_by_zone(boxes, zone_polygon):
    if zone_polygon is None or len(zone_polygon) < 3:
        return list(range(len(boxes)))
    inside = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Check multiple sample points — any overlap counts
        sample_points = [
            ((x1 + x2) / 2, y2),       # bottom-center (foot)
            ((x1 + x2) / 2, y1),       # top-center (head)
            (x1, (y1 + y2) / 2),       # left-center
            (x2, (y1 + y2) / 2),       # right-center
            ((x1 + x2) / 2, (y1 + y2) / 2),  # center
        ]
        for px, py in sample_points:
            if cv2.pointPolygonTest(zone_polygon, (float(px), float(py)), False) >= 0:
                inside.append(i)
                break
    return inside


# ---------------------------------------------------------------------------
# CameraState
# ---------------------------------------------------------------------------

class CameraState:
    def __init__(self, cam_id: str, source, label: str = ""):
        self.cam_id          = cam_id
        self.source          = source
        self.label           = label or f"Camera {cam_id}"
        self.session_id      = str(uuid.uuid4())[:8]
        self.trigger_restart = False
        self.active          = True
        self.paused          = False
        self.streaming       = False  # True while a generator is actively running
        self.zone_polygon    = None   # np.array (N,2) int32 in processing-frame pixel coords
        self.frame_w         = None
        self.frame_h         = None
        self.proc_w          = None
        self.proc_h          = None


# ---------------------------------------------------------------------------
# CameraRegistry
# ---------------------------------------------------------------------------

class CameraRegistry:
    def __init__(self):
        self._lock    = threading.Lock()
        self._cameras: dict[str, CameraState] = {}

    def add(self, source, label: str = "") -> CameraState:
        cam_id = str(uuid.uuid4())[:8]
        state  = CameraState(cam_id, source, label)
        with self._lock:
            self._cameras[cam_id] = state
        logger.info(f"[Registry] Camera added: {cam_id} → {source}")
        return state

    def remove(self, cam_id: str) -> bool:
        with self._lock:
            state = self._cameras.get(cam_id)
            if state:
                state.active = False
                del self._cameras[cam_id]
                return True
        return False

    def get(self, cam_id: str) -> CameraState | None:
        with self._lock:
            return self._cameras.get(cam_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [
                {"cam_id": s.cam_id, "label": s.label, "source": str(s.source)}
                for s in self._cameras.values()
            ]

    def clear(self):
        with self._lock:
            for s in self._cameras.values():
                s.active = False
            self._cameras.clear()


camera_registry = CameraRegistry()
stream_state    = StreamState()


# ---------------------------------------------------------------------------
# Inference worker (one per camera)
# ---------------------------------------------------------------------------

def _inference_worker(cam_state: CameraState, frame_q: queue.Queue, result_q: queue.Queue):
    tracker = sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=600,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=2,
    )
    ppe_ema, box_ema, ppe_state      = {}, {}, {}
    classify_lock                    = {}   # {tid: {state, started, locked_at, verdict}}
    violation_timer, last_seen_timer = {}, {}
    frame_count       = 0
    last_tracked      = sv.Detections.empty()
    last_ppe_results  = []
    last_pose_results = None
    last_cam_frame    = None

    while True:
        item = frame_q.get()
        if item is _STOP:
            result_q.put(_STOP)
            break

        frame, current_time = item
        frame_count += 1
        img_h, img_w = frame.shape[:2]

        has_zone = cam_state.zone_polygon is not None and len(cam_state.zone_polygon) >= 3
        if not has_zone:
            last_tracked = sv.Detections.empty()
            last_ppe_results = []
            result_q.put((frame.copy(), last_tracked, last_ppe_results))
            continue

        if frame_count % DETECT_EVERY == 0:
            res1 = model1.predict(frame, conf=CONF_M1, iou=IOU_NMS, device=device, imgsz=1280, verbose=False)
            res2 = model2.predict(frame, conf=CONF_M2, iou=IOU_NMS, device=device, verbose=False, classes=[0], imgsz=1280)
            b1, s1, l1 = extract_boxes(res1, img_w, img_h)
            b2, s2, l2 = extract_boxes(res2, img_w, img_h, class_filter=[0])

            if WBF_AVAILABLE:
                fused_boxes, fused_scores = fuse_boxes(b1, s1, l1, b2, s2, l2, img_w, img_h)
            else:
                fused_boxes  = np.array(b1) * np.array([img_w, img_h, img_w, img_h]) if b1 else np.empty((0, 4))
                fused_scores = np.array(s1, dtype=float) if b1 else np.array([])

            if len(fused_boxes) > 0:
                tracked = tracker.update_with_detections(
                    sv.Detections(
                        xyxy=fused_boxes.astype(float),
                        confidence=fused_scores,
                        class_id=np.zeros(len(fused_boxes), dtype=int),
                    )
                )
            else:
                tracked = sv.Detections.empty()

            if len(tracked) > 0 and tracked.tracker_id is not None:
                smoothed = []
                for tid, box in zip(tracked.tracker_id, tracked.xyxy):
                    box_ema[tid] = box if tid not in box_ema else BOX_EMA_ALPHA * box + (1 - BOX_EMA_ALPHA) * box_ema[tid]
                    smoothed.append(box_ema[tid])
                tracked.xyxy = np.array(smoothed)

                # Zone filter
                zone_ids = _filter_by_zone(tracked.xyxy, cam_state.zone_polygon)
                zone_boxes = tracked.xyxy[zone_ids] if zone_ids else tracked.xyxy[:0]
                zone_tids  = tracked.tracker_id[zone_ids] if zone_ids else tracked.tracker_id[:0]

                frame_clean = frame.copy()

                # ── Detect → Lock → Cooldown ──
                raw_results = [None] * len(tracked.xyxy)
                active_box_list = []
                active_zone_map = []   # (j_in_zone, zi_in_tracked)

                for j, zi in enumerate(zone_ids):
                    tid = int(zone_tids[j])
                    if tid not in classify_lock:
                        classify_lock[tid] = {
                            "state": "ACTIVE", "started": current_time,
                            "locked_at": 0.0, "verdict": None,
                            "candidate_sig": None, "candidate_count": 0,
                        }
                    lock = classify_lock[tid]

                    if lock["state"] == "ACTIVE":
                        elapsed = current_time - lock["started"]
                        consensus_ready = int(lock.get("candidate_count", 0)) >= int(CLASSIFY_LOCK_MIN_CONSISTENT)
                        if elapsed >= CLASSIFY_ACTIVE_SEC and consensus_ready and lock["verdict"] is not None and _is_verdict_stable(lock["verdict"]):
                            lock["state"] = "LOCKED"
                            lock["locked_at"] = current_time
                            raw_results[zi] = lock["verdict"]
                        else:
                            active_box_list.append(zone_boxes[j])
                            active_zone_map.append((j, zi))
                    elif lock["state"] == "LOCKED":
                        lock_age = current_time - lock["locked_at"]
                        unstable = not _is_verdict_stable(lock["verdict"])
                        if lock_age >= CLASSIFY_FORCE_RECHECK_SEC or unstable:
                            lock["state"] = "ACTIVE"
                            lock["started"] = current_time
                            lock["verdict"] = None
                            lock["candidate_sig"] = None
                            lock["candidate_count"] = 0
                            active_box_list.append(zone_boxes[j])
                            active_zone_map.append((j, zi))
                        else:
                            raw_results[zi] = lock["verdict"]

                # Classify only ACTIVE trackers
                if active_box_list:
                    active_boxes_arr = np.array(active_box_list)
                    active_tids_arr  = np.array([int(zone_tids[j]) for j, _ in active_zone_map])

                    pose_results = None
                    if should_run_pose_for_boxes(model_stage2, active_boxes_arr):
                        if frame_count % POSE_EVERY == 0 or last_pose_results is None:
                            last_pose_results = model_pose.predict(frame, conf=0.3, device=device, verbose=False)
                        pose_results = last_pose_results

                    active_raw = classify_ppe_batch(
                        model_stage2, frame, active_boxes_arr, device,
                        tracker_ids=active_tids_arr, pose_results=pose_results,
                    )
                    for k, (j, zi) in enumerate(active_zone_map):
                        raw_results[zi] = active_raw[k]
                        lock_rec = classify_lock[int(zone_tids[j])]
                        lock_rec["verdict"] = active_raw[k]
                        _update_lock_consensus(lock_rec, active_raw[k])

                ppe_results = update_ema_and_decide(
                    raw_results, tracked.tracker_id, tracked.xyxy,
                    frame_clean, ppe_ema, ppe_state, violation_timer,
                    current_time, cam_state.session_id,
                )
                garbage_collection(tracked.tracker_id, current_time, last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer, classify_lock, session_id=cam_state.session_id)
            else:
                ppe_results = []
                garbage_collection([], current_time, last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer, session_id=cam_state.session_id)

            last_tracked     = tracked
            last_ppe_results = ppe_results

            from src.inference.classifier import cam_mode_enabled
            last_cam_frame = frame.copy() if cam_mode_enabled else None

        else:
            tracked     = last_tracked
            ppe_results = last_ppe_results
            from src.inference.classifier import cam_mode_enabled
            if cam_mode_enabled and last_cam_frame is not None:
                np.copyto(frame, last_cam_frame)

        result_q.put((frame.copy(), tracked, ppe_results))


# ---------------------------------------------------------------------------
# Frame generator (one per camera stream client)
# ---------------------------------------------------------------------------

def generate_frames_for_camera(cam_id: str):
    cam_state = camera_registry.get(cam_id)
    if cam_state is None:
        logger.error(f"Unknown cam_id={cam_id}")
        return

    # Prevent multiple concurrent generators for the same camera.
    if cam_state.streaming:
        logger.warning(f"[{cam_id}] Generator already running, skipping duplicate request")
        return
    cam_state.streaming = True

    frame_q  = queue.Queue(maxsize=_FRAME_QUEUE_SIZE)
    result_q = queue.Queue(maxsize=_RESULT_QUEUE_SIZE)

    inf_thread = threading.Thread(
        target=_inference_worker,
        args=(cam_state, frame_q, result_q),
        daemon=True,
        name=f"inference-{cam_id}",
    )
    inf_thread.start()

    cap = None
    is_webcam = False

    def _open_cap():
        nonlocal cap, is_webcam
        if cap is not None:
            cap.release()
            cap = None
        source = cam_state.source
        logger.info(f"[{cam_id}] Opening: {source}")
        is_webcam = isinstance(source, int) or (isinstance(source, str) and str(source).isdigit())
        if is_webcam:
            # Avoid CAP_DSHOW — it raises unrecoverable C++ exceptions on some
            # Windows/camera combinations that can crash the entire process.
            # CAP_MSMF (Windows Media Foundation) is stable on Windows 8+.
            cap = cv2.VideoCapture(int(source), cv2.CAP_MSMF)
            time.sleep(1.0)
            if not cap.isOpened():
                cap.release()
                # Last resort: let OpenCV pick any available backend.
                cap = cv2.VideoCapture(int(source))
                time.sleep(1.0)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"[{cam_id}] Failed to open source")
            return False
        return True

    try:
        if not _open_cap():
            return

        while cam_state.active:
            # ---- PAUSED: freeze on last cached frame ----
            if cam_state.paused:
                cached = _latest_frames.get(cam_id)
                if cached:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cached + b'\r\n')
                time.sleep(0.05)
                continue

            if cam_state.trigger_restart:
                cam_state.trigger_restart = False
                if not _open_cap():
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                logger.info(f"[{cam_id}] EOF")
                if is_webcam:
                    logger.info(f"[{cam_id}] Webcam EOF — attempting reconnect in 2s")
                    time.sleep(2.0)
                    _open_cap()
                    continue
                break

            try:
                frame_q.put((frame.copy(), time.time()), timeout=1.0)
            except queue.Full:
                continue

            try:
                item = result_q.get(timeout=2.0)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            proc_frame, tracked, ppe_results = item

            from src.inference import draw_ppe_result
            annotated    = proc_frame
            img_h, img_w = annotated.shape[:2]
            cam_state.proc_w = img_w
            cam_state.proc_h = img_h

            # Zone overlay
            if cam_state.zone_polygon is not None and len(cam_state.zone_polygon) >= 3:
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [cam_state.zone_polygon], (56, 139, 253))
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                cv2.polylines(annotated, [cam_state.zone_polygon], True, (56, 139, 253), 2, cv2.LINE_AA)
            else:
                cv2.putText(
                    annotated,
                    "Set zone to start inference",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )

            if len(tracked) > 0 and tracked.tracker_id is not None:
                for box, tid, ppe_result in zip(tracked.xyxy, tracked.tracker_id, ppe_results):
                    draw_ppe_result(annotated, box, ppe_result, tid)

            # Camera label
            cv2.putText(annotated, cam_state.label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            scale        = 720 / img_h
            target_width = int(img_w * scale)
            display      = cv2.resize(annotated, (target_width, 720))

            ret2, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret2:
                continue

            _latest_frames[cam_id] = buffer.tobytes()
            cam_state.frame_w      = target_width
            cam_state.frame_h      = 720

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        cam_state.streaming = False
        frame_q.put(_STOP)
        inf_thread.join(timeout=5.0)
        if cap is not None:
            cap.release()
        logger.info(f"[{cam_id}] Stream closed.")


# ---------------------------------------------------------------------------
# Legacy single-camera support
# ---------------------------------------------------------------------------

def generate_frames():
    if stream_state.source is None:
        return
    existing = next(
        (s for s in camera_registry._cameras.values() if str(s.source) == str(stream_state.source)),
        None,
    )
    if existing is None or stream_state.trigger_restart:
        if existing is not None:
            camera_registry.remove(existing.cam_id)
        cam_state = camera_registry.add(source=stream_state.source, label="Zone A")
        stream_state.trigger_restart = False
        stream_state.session_id      = cam_state.session_id
    else:
        cam_state = existing
    yield from generate_frames_for_camera(cam_state.cam_id)