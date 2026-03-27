import cv2
import time
import logging
from src.config import (
    LABEL_COLS, PPE_THRESHOLDS, HYSTERESIS_MARGIN,
    USE_EMA, EMA_ALPHA, ALERT_TIME_THRESHOLD,
    GRACE_PERIOD_SEC, STATE_COLORS, COLOR_TRACK,
)
from src.infrastructure.supabase import log_violation_to_supabase, db_executor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Violation tracking — replaces reported_ids set
# {tid: {
#   "violations": set(),          # Already reported violations
#   "last_reported": float,        # Last time reported to supabase
#   "cached_items": set(),         # Items waiting in cache (only 1/2)
#   "cache_start_time": float,     # When cache started
# }}
# ---------------------------------------------------------------------------
reported_violations: dict = {}

VIOLATION_COOLDOWN_SEC = 300.0  # 5 minutes before re-reporting same tracker
SINGLE_ITEM_CACHE_SEC = 10.0    # Cache timeout for single missing item (short time)


class StreamState:
    def __init__(self):
        self.source          = None
        self.trigger_restart = False
        self.session_id      = None


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


def _should_report(tid, missing_items: list, current_time: float) -> list:
    """
    Returns list of violation types that should be logged to supabase.
    
    Caching logic for single-item violations:
    - If only 1 out of 2 items missing: cache it, don't report yet
    - If cache timeout expires and still 1 missing: report cached items
    - If 2 items missing (escalation): report all immediately
    - If back to all OK: clear cache
    """
    if tid not in reported_violations:
        reported_violations[tid] = {
            "violations":       set(),
            "last_reported":    0.0,
            "cached_items":     set(),
            "cache_start_time": 0.0,
        }

    rec = reported_violations[tid]
    missing_set = set(missing_items)

    # Cooldown expired → reset reported violations, but keep cache
    if current_time - rec["last_reported"] >= VIOLATION_COOLDOWN_SEC:
        rec["violations"]    = set()
        rec["last_reported"] = 0.0

    # If no violations now, clear cache
    if not missing_set:
        rec["cached_items"]     = set()
        rec["cache_start_time"] = 0.0
        return []

    # Find new violations not yet reported in this cooldown window
    new_items = missing_set - rec["violations"]

    # Handle single-item (1 out of 2) violations
    if len(missing_set) == 1:
        # If different item than cached, reset and start caching new item
        if missing_set != rec["cached_items"]:
            rec["cached_items"] = missing_set.copy()
            rec["cache_start_time"] = current_time
            return []  # Don't report yet, just cache
        
        # Same item - check if timeout expired
        cache_age = current_time - rec["cache_start_time"]
        if cache_age >= SINGLE_ITEM_CACHE_SEC:
            # Cache timeout reached - report the cached items
            to_report = rec["cached_items"] - rec["violations"]
            if to_report:
                logger.info(f"[TID {tid}] Cache timeout ({cache_age:.1f}s >= {SINGLE_ITEM_CACHE_SEC}s) - reporting {to_report}")
                rec["violations"].update(to_report)
                rec["last_reported"] = current_time
                return list(to_report)
        else:
            logger.debug(f"[TID {tid}] Single item {missing_set} cached ({cache_age:.1f}s < {SINGLE_ITEM_CACHE_SEC}s)")
        
        # Still within cache timeout - don't report yet
        return []

    # Handle multiple-item (2 out of 2) violations - escalation
    elif len(missing_set) >= 2:
        # Clear cache since we now have escalated violation
        logger.info(f"[TID {tid}] ESCALATION to {missing_set} - reporting new items {new_items}")
        rec["cached_items"]     = set()
        rec["cache_start_time"] = 0.0
        
        # Report any new items to supabase
        if new_items:
            rec["violations"].update(new_items)
            rec["last_reported"] = current_time
            return list(new_items)
    
    return []


def update_ema_and_decide(
    raw_results, tracker_ids, boxes, frame,
    ppe_ema, ppe_state, violation_timer, current_time,
    session_id=None,
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

        quality = result.get("quality", {}) if isinstance(result, dict) else {}
        if quality.get("hardhat_visibility_low", False) and len(new_states) > 0 and new_states[0] == 0:
            # Low-visibility head should not be escalated directly to hardhat miss.
            new_states[0] = 1

        if tid not in violation_timer:
            violation_timer[tid] = {"duration": 0.0, "last_tick": current_time}

        elapsed = current_time - violation_timer[tid]["last_tick"]
        violation_timer[tid]["last_tick"] = current_time

        if min(new_states) < 2:
            violation_timer[tid]["duration"] += elapsed
        else:
            violation_timer[tid]["duration"] = 0.0

        if violation_timer[tid]["duration"] >= ALERT_TIME_THRESHOLD:
            missing_items = [LABEL_COLS[i] for i, s in enumerate(new_states) if s == 0]
            missing_probs = [avg_probs[i] for i, s in enumerate(new_states) if s == 0]

            if missing_items:
                to_report = _should_report(tid, missing_items, current_time)
                if to_report:
                    report_probs = [
                        avg_probs[LABEL_COLS.index(item)]
                        for item in to_report
                    ]
                    reporter_model = str(result.get("router_model", "unknown")).lower() if isinstance(result, dict) else "unknown"
                    import numpy as np
                    img_h, img_w = frame.shape[:2]
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    crop = frame[y1:y2, x1:x2].copy()
                    db_executor.submit(
                        log_violation_to_supabase,
                        tid, to_report, report_probs, crop, session_id, reporter_model,
                    )

        ppe_state[tid] = new_states
        smoothed.append({
            "probs": avg_probs,
            "states": new_states,
            "router_model": result.get("router_model") if isinstance(result, dict) else None,
        })

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
        reported_violations.pop(k, None)
        del last_seen_timer[k]


def draw_ppe_result(frame, box, ppe_result, tracker_id):
    """Renders bounding boxes and probabilistic telemetry on the active frame."""
    x1, y1, x2, y2 = map(int, box)
    if ppe_result is None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TRACK, 2)
        cv2.putText(frame, f"ID:{tracker_id}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TRACK, 1)
        return

    states     = ppe_result["states"]
    min_state  = min(states)
    main_color = STATE_COLORS[min_state]

    label_parts = []
    for i, state in enumerate(states):
        name = LABEL_COLS[i][:4].upper()
        if state == 0:   label_parts.append(f"MISS {name}")
        elif state == 1: label_parts.append(f"WARN {name}")

    model_tag = ""
    if isinstance(ppe_result, dict) and ppe_result.get("router_model"):
        model_tag = f" [{str(ppe_result['router_model']).upper()}]"
    label_text = f"ID:{tracker_id}{model_tag} " + ("- ".join(label_parts) if label_parts else "ALL OK")

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