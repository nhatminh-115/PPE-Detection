import time
import logging
import queue
import threading
import numpy as np
import supervision as sv
from src.api.app import app, model1, model2, model_pose, model_stage2, device
from src.config import CONF_M1, CONF_M2, IOU_NMS, DETECT_EVERY, BOX_EMA_ALPHA
from src.inference import (
    extract_boxes, fuse_boxes, WBF_AVAILABLE,
    classify_ppe_batch, update_ema_and_decide,
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


# ---------------------------------------------------------------------------
# Zone filtering
# ---------------------------------------------------------------------------

def _filter_by_zone(boxes, zone_polygon):
    if zone_polygon is None or len(zone_polygon) < 3:
        return list(range(len(boxes)))
    inside = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        foot_x = float((x1 + x2) / 2)
        foot_y = float(y2)
        if cv2.pointPolygonTest(zone_polygon, (foot_x, foot_y), False) >= 0:
            inside.append(i)
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
        lost_track_buffer=240,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=2,
    )
    ppe_ema, box_ema, ppe_state      = {}, {}, {}
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
            if frame_count % POSE_EVERY == 0:
                last_pose_results = model_pose.predict(frame, conf=0.3, device=device, verbose=False)
            pose_results = last_pose_results

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
                raw_zone = classify_ppe_batch(
                    model_stage2, frame, zone_boxes, device,
                    tracker_ids=zone_tids, pose_results=pose_results,
                )
                raw_results = [None] * len(tracked.xyxy)
                for i, zi in enumerate(zone_ids):
                    raw_results[zi] = raw_zone[i]

                ppe_results = update_ema_and_decide(
                    raw_results, tracked.tracker_id, tracked.xyxy,
                    frame_clean, ppe_ema, ppe_state, violation_timer,
                    current_time, cam_state.session_id,
                )
                garbage_collection(tracked.tracker_id, current_time, last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer)
            else:
                ppe_results = []
                garbage_collection([], current_time, last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer)

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

    def _open_cap():
        nonlocal cap
        if cap is not None:
            cap.release()
        source = cam_state.source
        logger.info(f"[{cam_id}] Opening: {source}")
        if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            time.sleep(0.5)
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