import cv2
import time
import logging
from src.config import (
    LABEL_COLS, PPE_THRESHOLDS, HYSTERESIS_MARGIN,
    USE_EMA, EMA_ALPHA, ALERT_TIME_THRESHOLD,
    GRACE_PERIOD_SEC, STATE_COLORS, COLOR_TRACK, VIOLATION_COOLDOWN_SEC
)
from src.infrastructure.supabase import log_violation_to_supabase, db_executor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Violation tracking — per-item cooldown
# {tid: {item_name: last_reported_time}}
# Each PPE item (hardhat, vest) tracks its own 5-min cooldown independently.
# Full recovery (all PPE back) clears all cooldowns for that tracker.
# ---------------------------------------------------------------------------
reported_violations: dict = {}


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


def _should_report(tid, missing_items: list, current_time: float, session_id=None) -> list:
    """
    Returns list of violation types that should be logged to supabase.

    Cooldown semantics (per-item, not per-event):
    - Each PPE item has its own independent 5-min cooldown.
    - An item is reported when it first goes missing; re-reported only after
      VIOLATION_COOLDOWN_SEC has elapsed since its last report.
    - Full recovery (all items OK) clears all per-item cooldowns so the next
      violation event starts fresh.
    - Partial recovery (e.g. vest back, hardhat still off) does NOT reset
      hardhat's cooldown — it was already reported and the timer stands.
      This prevents oscillating EMA states from spamming logs.
    - Key is (session_id, tid) to avoid cross-camera tid collisions.
    """
    key = (session_id, tid)
    if key not in reported_violations:
        reported_violations[key] = {}   # {item_name: last_reported_time}

    rec = reported_violations[key]
    missing_set = set(missing_items)

    to_report = []
    for item in missing_set:
        last_time = rec.get(item, 0.0)
        if last_time == 0.0 or current_time - last_time >= VIOLATION_COOLDOWN_SEC:
            to_report.append(item)

    if to_report:
        logger.info(f"[SID {session_id} TID {tid}] Reporting violation: {to_report} (missing: {missing_set})")
        for item in to_report:
            rec[item] = current_time

    return to_report


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
            # Only hold at WARN when prob is still in the ambiguous range.
            # If EMA has clearly settled near 0, the person is genuinely not wearing
            # a hardhat — do not suppress the MISSING state.
            warn_th = float(PPE_THRESHOLDS["hardhat"]["warn"])
            if float(avg_probs[0]) >= warn_th * 0.35:
                new_states[0] = 1

        if tid not in violation_timer:
            violation_timer[tid] = {"last_tick": current_time, "durations": {}}

        vt = violation_timer[tid]
        elapsed = current_time - vt["last_tick"]
        vt["last_tick"] = current_time

        # Per-item timers: each PPE item accumulates independently.
        # A brief dip on one item cannot piggyback on another item's elapsed time.
        for i, label in enumerate(LABEL_COLS):
            if new_states[i] == 0:
                vt["durations"][label] = vt["durations"].get(label, 0.0) + elapsed
            else:
                vt["durations"][label] = 0.0

        ready_items = [
            label for label in LABEL_COLS
            if vt["durations"].get(label, 0.0) >= ALERT_TIME_THRESHOLD
        ]

        # Full recovery: ALL items fully OK (state=2, green) → clear per-item
        # cooldowns so the next violation event is treated as fresh.
        # WARN (state=1) is NOT a recovery — EMA oscillation through WARN must
        # not reset cooldowns or it will spam on every brief ambiguous frame.
        if all(s == 2 for s in new_states):
            reported_violations.pop((session_id, tid), None)

        if ready_items:
            to_report = _should_report(tid, ready_items, current_time, session_id)
            if to_report:
                report_probs = [avg_probs[LABEL_COLS.index(item)] for item in to_report]
                reporter_model = str(result.get("router_model", "unknown")).lower() if isinstance(result, dict) else "unknown"
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


def garbage_collection(active_ids, current_time, last_seen_timer, *state_dicts, session_id=None):
    """Executes state cleanup utilizing Time-To-Live (TTL) grace periods."""
    for tid in active_ids:
        last_seen_timer[tid] = current_time

    stale_keys = [k for k, t in last_seen_timer.items() if current_time - t > GRACE_PERIOD_SEC]
    for k in stale_keys:
        for d in state_dicts:
            if k in d:
                del d[k]
        reported_violations.pop((session_id, k), None)
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