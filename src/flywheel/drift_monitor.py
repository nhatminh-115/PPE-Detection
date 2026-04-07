"""
Phase 4 — Scout Disagreement Rate Monitor

Computes daily disagreement signals and writes them to the drift_log table.

Signals:
    - Scout disagreement rate: model vs Scout on auto-labeled events (early warning)
    - Human disagreement rate: model vs human-confirmed labels (retrain trigger source)

Retrain trigger:
    7-day rolling average of human_disagreement_rate > DRIFT_THRESHOLD (default 0.25)
  Minimum 3 days of data required before a trigger fires.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD        = float(os.environ.get("DRIFT_THRESHOLD", "0.25"))
MIN_CONFIRMED_SAMPLES  = int(os.environ.get("MIN_CONFIRMED_SAMPLES", "100"))
ROLLING_WINDOW         = 7   # days
MIN_DATA_DAYS          = 3   # guard against early false triggers
ICT = timezone(timedelta(hours=7))


def compute_daily_rate(date_str: str) -> dict:
    """
    Compute daily Scout and Human disagreement rates for date_str.
    """
    from src.infrastructure.supabase import get_supabase_service_client

    client = get_supabase_service_client()
    if not client:
        raise RuntimeError("Supabase service client unavailable")

    day_start = f"{date_str}T00:00:00+07:00"
    day_end   = f"{date_str}T23:59:59+07:00"

    scout_by_key: dict[str, dict] = {}

    # Prefer immutable audit trail so Scout signal is preserved even after human overwrite.
    try:
        audit_rows = (
            client.table("ppe_label_audit")
            .select("merge_key, created_at, new_state")
            .gte("created_at", day_start)
            .lt("created_at", day_end)
            .order("created_at")
            .execute()
            .data or []
        )

        for row in audit_rows:
            state = row.get("new_state") or {}
            if not isinstance(state, dict):
                continue
            if not bool(state.get("is_auto_labeled", False)):
                continue
            key = str(row.get("merge_key") or state.get("merge_key") or "")
            if not key:
                continue
            scout_by_key[key] = {
                "aggregate": state.get("aggregate"),
                "labeled_at": state.get("labeled_at"),
            }
    except Exception as exc:
        logger.warning("drift_monitor: audit read failed, fallback to ppe_labels: %s", exc)

    # Fallback for older data where scout writes were not audited.
    if not scout_by_key:
        rows = (
            client.table("ppe_labels")
            .select("merge_key, aggregate")
            .eq("is_auto_labeled", True)
            .gte("labeled_at", day_start)
            .lt("labeled_at", day_end)
            .execute()
            .data or []
        )
        for row in rows:
            key = str(row.get("merge_key") or "")
            if not key:
                continue
            scout_by_key[key] = {"aggregate": row.get("aggregate")}

    scout_total_auto = len(scout_by_key)
    scout_disagreements = sum(
        1 for r in scout_by_key.values() if r.get("aggregate") in ("has-fp", "has-fn")
    )
    scout_rate = round(scout_disagreements / scout_total_auto, 4) if scout_total_auto > 0 else 0.0

    # Human-confirmed signal (ground truth path used for retrain trigger)
    human_rows = (
        client.table("ppe_labels")
        .select("merge_key, aggregate")
        .eq("is_auto_labeled", False)
        .eq("skipped", False)
        .gte("labeled_at", day_start)
        .lt("labeled_at", day_end)
        .execute()
        .data or []
    )
    human_by_key: dict[str, str] = {}
    for row in human_rows:
        key = str(row.get("merge_key") or "")
        if not key:
            continue
        human_by_key[key] = str(row.get("aggregate") or "")

    human_total_confirmed = len(human_by_key)
    human_disagreements = sum(1 for v in human_by_key.values() if v in ("has-fp", "has-fn"))
    human_rate = round(human_disagreements / human_total_confirmed, 4) if human_total_confirmed > 0 else 0.0

    return {
        "date":                      date_str,
        "scout_total_auto":          scout_total_auto,
        "scout_disagreements":       scout_disagreements,
        "scout_disagreement_rate":   scout_rate,
        "human_total_confirmed":     human_total_confirmed,
        "human_disagreements":       human_disagreements,
        "human_disagreement_rate":   human_rate,
        # Canonical disagreement signal for retrain = human-confirmed.
        "disagreement_rate":         human_rate,
        # Backward-compatible keys kept for callers that still expect these names.
        "total_auto":                scout_total_auto,
        "disagreements":             scout_disagreements,
    }


def _rolling_rates_from_log(date_str: str) -> tuple[float | None, float | None]:
    """
    Return (human_rolling, scout_rolling) for the last ROLLING_WINDOW rows.
    Human rolling falls back to legacy disagreement_rate when human_disagreement_rate
    is unavailable in older rows.
    """
    from src.infrastructure.supabase import get_supabase_service_client

    client = get_supabase_service_client()
    if not client:
        return None

    window_start = (
        datetime.strptime(date_str, "%Y-%m-%d").date()
        - timedelta(days=ROLLING_WINDOW - 1)
    ).isoformat()

    try:
        rows = (
            client.table("drift_log")
            .select("date, disagreement_rate, human_disagreement_rate, scout_disagreement_rate")
            .gte("date", window_start)
            .lte("date", date_str)
            .order("date")
            .execute()
            .data or []
        )
    except Exception:
        # Backward compatibility before migration 007 is applied.
        rows = (
            client.table("drift_log")
            .select("date, disagreement_rate")
            .gte("date", window_start)
            .lte("date", date_str)
            .order("date")
            .execute()
            .data or []
        )

    human_rates = []
    scout_rates = []
    for r in rows:
        hr = r.get("human_disagreement_rate")
        if hr is None:
            hr = r.get("disagreement_rate")
        if hr is not None:
            human_rates.append(hr)

        sr = r.get("scout_disagreement_rate")
        if sr is not None:
            scout_rates.append(sr)

    human_rolling = None
    scout_rolling = None
    if len(human_rates) >= MIN_DATA_DAYS:
        human_rolling = round(sum(human_rates) / len(human_rates), 4)
    if len(scout_rates) >= MIN_DATA_DAYS:
        scout_rolling = round(sum(scout_rates) / len(scout_rates), 4)

    return human_rolling, scout_rolling


def _count_confirmed_labels() -> int:
    """Return total confirmed (is_auto_labeled=False, skipped=False) label count."""
    from src.infrastructure.supabase import get_supabase_service_client
    client = get_supabase_service_client()
    if not client:
        return 0
    try:
        res = (
            client.table("ppe_labels")
            .select("id", count="exact")
            .eq("is_auto_labeled", False)
            .eq("skipped", False)
            .execute()
        )
        return res.count or 0
    except Exception as exc:
        logger.warning("drift_monitor: confirmed label count failed: %s", exc)
        return 0


def run_drift_monitor(date_str: str | None = None) -> dict:
    """
    Compute daily disagreement rate, update drift_log, and return drift stats
    including whether EfficientNet retrain should be triggered.

    Retrain triggers only when BOTH conditions are met:
      - rolling_7d_rate > DRIFT_THRESHOLD
      - total confirmed labels >= MIN_CONFIRMED_SAMPLES
    """
    from src.infrastructure.supabase import get_supabase_service_client

    if date_str is None:
        date_str = datetime.now(ICT).strftime("%Y-%m-%d")

    daily                   = compute_daily_rate(date_str)
    rolling_human, rolling_scout = _rolling_rates_from_log(date_str)
    confirmed_total         = _count_confirmed_labels()

    rate_ok    = rolling_human is not None and rolling_human > DRIFT_THRESHOLD
    samples_ok = confirmed_total >= MIN_CONFIRMED_SAMPLES
    retrain_triggered = rate_ok and samples_ok

    if rate_ok and not samples_ok:
        logger.info(
            "drift_monitor: rate threshold met (%.4f) but only %d/%d confirmed samples — retrain deferred",
            rolling_human, confirmed_total, MIN_CONFIRMED_SAMPLES,
        )

    client = get_supabase_service_client()
    if client:
        try:
            payload = {
                "date":                    date_str,
                "total_auto":              daily["total_auto"],
                "disagreements":           daily["disagreements"],
                # Canonical disagreement for trigger/reporting = human rate.
                "disagreement_rate":       daily["human_disagreement_rate"],
                "rolling_7d_rate":         rolling_human,
                "scout_total_auto":        daily["scout_total_auto"],
                "scout_disagreements":     daily["scout_disagreements"],
                "scout_disagreement_rate": daily["scout_disagreement_rate"],
                "human_total_confirmed":   daily["human_total_confirmed"],
                "human_disagreements":     daily["human_disagreements"],
                "human_disagreement_rate": daily["human_disagreement_rate"],
                "rolling_7d_rate_scout":   rolling_scout,
                "retrain_triggered":       retrain_triggered,
                "computed_at":             datetime.now(timezone.utc).isoformat(),
            }
            try:
                client.table("drift_log").upsert(payload, on_conflict="date").execute()
            except Exception:
                # Backward compatibility when extra columns are not migrated yet.
                client.table("drift_log").upsert(
                    {
                        "date":              date_str,
                        "total_auto":        daily["human_total_confirmed"],
                        "disagreements":     daily["human_disagreements"],
                        "disagreement_rate": daily["human_disagreement_rate"],
                        "rolling_7d_rate":   rolling_human,
                        "retrain_triggered": retrain_triggered,
                        "computed_at":       datetime.now(timezone.utc).isoformat(),
                    },
                    on_conflict="date",
                ).execute()
        except Exception as exc:
            logger.warning("drift_monitor: drift_log upsert failed: %s", exc)

    result = {
        **daily,
        "rolling_7d_rate":        rolling_human,
        "rolling_7d_rate_scout":  rolling_scout,
        "drift_threshold":        DRIFT_THRESHOLD,
        "retrain_triggered":      retrain_triggered,
    }
    logger.info("drift_monitor: %s", result)
    return result
