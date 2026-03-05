import cv2
from src.config import (
    LABEL_COLS, PPE_THRESHOLDS, HYSTERESIS_MARGIN,
    USE_EMA, EMA_ALPHA, ALERT_TIME_THRESHOLD,
    GRACE_PERIOD_SEC, STATE_COLORS, COLOR_TRACK,
)
from src.infrastructure.supabase import log_violation_to_supabase, db_executor, reported_ids


class StreamState:
    def __init__(self):
        self.source = None
        self.trigger_restart = False


def get_next_state(class_name: str, prob: float, current_state) -> int:
    """Evaluates FSM transitions utilizing mathematical hysteresis."""
    th = PPE_THRESHOLDS[class_name]
    if current_state is None:
        if prob >= th['ok']:    return 2
        if prob >= th['warn']:  return 1
        return 0
    if current_state == 2:
        if prob < th['ok'] - HYSTERESIS_MARGIN:
            return 1 if prob >= th['warn'] else 0
        return 2
    elif current_state == 1:
        if prob >= th['ok'] + HYSTERESIS_MARGIN:   return 2
        if prob < th['warn'] - HYSTERESIS_MARGIN:  return 0
        return 1
    else:
        if prob >= th['warn'] + HYSTERESIS_MARGIN:
            return 2 if prob >= th['ok'] + HYSTERESIS_MARGIN else 1
        return 0


def update_ema_and_decide(
    raw_results, tracker_ids, boxes, frame,
    ppe_ema, ppe_state, violation_timer, current_time,
):
    """Processes temporal smoothing and orchestrates violation accumulation logic."""
    smoothed = []

    for tid, result, box in zip(tracker_ids, raw_results, boxes):
        if result is None:
            smoothed.append(None)
            continue

        curr_prob = result["probs"]
        if USE_EMA:
            if tid not in ppe_ema:
                ppe_ema[tid] = curr_prob.copy()
            else:
                ppe_ema[tid] = EMA_ALPHA * curr_prob + (1.0 - EMA_ALPHA) * ppe_ema[tid]
            avg_probs = ppe_ema[tid]
        else:
            avg_probs = curr_prob

        current_states = ppe_state.get(tid, [None] * len(LABEL_COLS))
        new_states = [
            get_next_state(LABEL_COLS[i], avg_probs[i], current_states[i])
            for i in range(len(LABEL_COLS))
        ]

        if tid not in violation_timer:
            violation_timer[tid] = {"duration": 0.0, "last_tick": current_time}

        elapsed = current_time - violation_timer[tid]["last_tick"]
        violation_timer[tid]["last_tick"] = current_time

        if min(new_states) < 2:
            violation_timer[tid]["duration"] += elapsed
        else:
            violation_timer[tid]["duration"] = 0.0

        if violation_timer[tid]["duration"] >= ALERT_TIME_THRESHOLD and tid not in reported_ids:
            reported_ids.add(tid)
            missing_items = [LABEL_COLS[i] for i, s in enumerate(new_states) if s == 0]
            missing_probs = [avg_probs[i] for i, s in enumerate(new_states) if s == 0]

            if missing_items:
                import numpy as np
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
    """Renders bounding boxes and probabilistic telemetry on the active frame."""
    x1, y1, x2, y2 = map(int, box)
    if ppe_result is None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TRACK, 2)
        cv2.putText(frame, f"ID:{tracker_id}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TRACK, 1)
        return

    states    = ppe_result["states"]
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
    cv2.putText(frame, label_text, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    bar_y = y2 + 4
    for i, (lbl, prob, state) in enumerate(zip(LABEL_COLS, ppe_result["probs"], states)):
        bar_color = STATE_COLORS[state]
        cv2.rectangle(frame, (x1, bar_y + i * 11),
                      (x1 + int(prob * 50), bar_y + i * 11 + 7), bar_color, -1)
        cv2.putText(frame, f"{lbl[:4]}:{prob:.2f}",
                    (x1 + 53, bar_y + i * 11 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)