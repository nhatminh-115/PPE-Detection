import time
import logging
import numpy as np
import supervision as sv
from src.api.app import app, model1, model2, model_pose, model_stage2, device, executor
from src.config import CONF_M1, CONF_M2, IOU_NMS, DETECT_EVERY, BOX_EMA_ALPHA
from src.inference import (
    extract_boxes, fuse_boxes, WBF_AVAILABLE,
    classify_ppe_batch, update_ema_and_decide,
    garbage_collection, StreamState,
)
import cv2

logger = logging.getLogger(__name__)

stream_state = StreamState()


def generate_frames():
    """Async frame generator with pose-guided PPE classification."""
    tracker = sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=240,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=2, 
    )

    last_tracked     = sv.Detections.empty()
    last_ppe_results = []
    ppe_ema, box_ema, ppe_state = {}, {}, {}
    violation_timer, last_seen_timer = {}, {}
    frame_count    = 0
    cap            = None
    last_cam_frame = None

    while True:
        if cap is None or stream_state.trigger_restart:
            if cap is not None:
                cap.release()
            if stream_state.source is None:
                time.sleep(1)
                continue
            logger.info(f"Establishing stream: {stream_state.source}")
            source = stream_state.source
            if isinstance(source, str) and source.isdigit():
                cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
                time.sleep(0.5)
            else:
                cap = cv2.VideoCapture(source)

            stream_state.trigger_restart = False
            if not cap.isOpened():
                logger.error("Failed to acquire video stream.")
                cap = None
                stream_state.source = None
                continue

        ret, frame = cap.read()
        if not ret:
            logger.info("Stream reached EOF.")
            cap.release()
            cap = None
            stream_state.source = None
            continue

        frame_count += 1
        img_h, img_w = frame.shape[:2]
        current_time = time.time()

        if frame_count % DETECT_EVERY == 0:
            # Run detection + pose in parallel
            future1 = executor.submit(
                model1.predict, frame,
                conf=CONF_M1, iou=IOU_NMS, device=device, imgsz=1280, verbose=False,
            )
            future2 = executor.submit(
                model2.predict, frame,
                conf=CONF_M2, iou=IOU_NMS, device=device,
                verbose=False, classes=[0], imgsz=1280,
            )
            future3 = executor.submit(
                model_pose.predict, frame,
                conf=0.3, device=device, verbose=False,
            )

            res1, res2, pose_results = future1.result(), future2.result(), future3.result()

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
                smoothed_boxes = []
                for tid, box in zip(tracked.tracker_id, tracked.xyxy):
                    if tid not in box_ema:
                        box_ema[tid] = box
                    else:
                        box_ema[tid] = BOX_EMA_ALPHA * box + (1.0 - BOX_EMA_ALPHA) * box_ema[tid]
                    smoothed_boxes.append(box_ema[tid])
                tracked.xyxy = np.array(smoothed_boxes)

                frame_clean = frame.copy()  # clean copy before CAM overlay
                raw_results = classify_ppe_batch(
                    model_stage2, frame, tracked.xyxy, device,
                    tracker_ids=tracked.tracker_id,
                    pose_results=pose_results,
                )
                ppe_results = update_ema_and_decide(
                    raw_results, tracked.tracker_id, tracked.xyxy,
                    frame_clean, ppe_ema, ppe_state, violation_timer, current_time,
                    stream_state.session_id,
                )
                garbage_collection(
                    tracked.tracker_id, current_time,
                    last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer,
                )

                from src.inference.classifier import cam_mode_enabled
                if cam_mode_enabled:
                    last_cam_frame = frame.copy()

            else:
                ppe_results = []
                garbage_collection(
                    [], current_time,
                    last_seen_timer, ppe_ema, box_ema, ppe_state, violation_timer,
                )

            last_tracked, last_ppe_results = tracked, ppe_results

        else:
            tracked, ppe_results = last_tracked, last_ppe_results
            from src.inference.classifier import cam_mode_enabled
            if cam_mode_enabled and last_cam_frame is not None and len(tracked) > 0:
                for box in tracked.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    if x2 > x1 and y2 > y1:
                        frame[y1:y2, x1:x2] = last_cam_frame[y1:y2, x1:x2]

        from src.inference import draw_ppe_result
        annotated = frame.copy()
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for box, tid, ppe_result in zip(tracked.xyxy, tracked.tracker_id, ppe_results):
                draw_ppe_result(annotated, box, ppe_result, tid)

        scale        = 720 / img_h
        target_width = int(img_w * scale)
        display      = cv2.resize(annotated, (target_width, 720))

        ret, buffer  = cv2.imencode('.jpg', display)
        frame_bytes  = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )