"""
Phase 4 — Scout Disagreement Rate Monitor

Computes the daily Scout disagreement rate from ppe_labels auto-labeled rows
and writes it to the drift_log table.  Returns the 7-day rolling average.

Drift metric:
  disagreement_rate = (has-fp + has-fn auto-labels) / total auto-labels for the day

Retrain trigger:
  7-day rolling average of disagreement_rate > DRIFT_THRESHOLD (default 0.25)
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
    Count Scout auto-labels for date_str and compute the raw disagreement_rate.
    """
    from src.infrastructure.supabase import get_supabase_service_client

    client = get_supabase_service_client()
    if not client:
        raise RuntimeError("Supabase service client unavailable")

    day_start = f"{date_str}T00:00:00+07:00"
    day_end   = f"{date_str}T23:59:59+07:00"

    rows = (
        client.table("ppe_labels")
        .select("aggregate")
        .eq("is_auto_labeled", True)
        .gte("labeled_at", day_start)
        .lt("labeled_at", day_end)
        .execute()
        .data or []
    )

    total_auto    = len(rows)
    disagreements = sum(
        1 for r in rows if r.get("aggregate") in ("has-fp", "has-fn")
    )
    rate = round(disagreements / total_auto, 4) if total_auto > 0 else 0.0

    return {
        "date":              date_str,
        "total_auto":        total_auto,
        "disagreements":     disagreements,
        "disagreement_rate": rate,
    }


def _rolling_rate_from_log(date_str: str) -> float | None:
    """
    Read the last ROLLING_WINDOW drift_log rows up to date_str and return
    their average disagreement_rate.  Returns None when fewer than MIN_DATA_DAYS
    records exist (avoids early false triggers).
    """
    from src.infrastructure.supabase import get_supabase_service_client

    client = get_supabase_service_client()
    if not client:
        return None

    window_start = (
        datetime.strptime(date_str, "%Y-%m-%d").date()
        - timedelta(days=ROLLING_WINDOW - 1)
    ).isoformat()

    rows = (
        client.table("drift_log")
        .select("date, disagreement_rate")
        .gte("date", window_start)
        .lte("date", date_str)
        .order("date")
        .execute()
        .data or []
    )

    rates = [r["disagreement_rate"] for r in rows if r.get("disagreement_rate") is not None]
    if len(rates) < MIN_DATA_DAYS:
        return None

    return round(sum(rates) / len(rates), 4)


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

    daily             = compute_daily_rate(date_str)
    rolling           = _rolling_rate_from_log(date_str)
    confirmed_total   = _count_confirmed_labels()

    rate_ok    = rolling is not None and rolling > DRIFT_THRESHOLD
    samples_ok = confirmed_total >= MIN_CONFIRMED_SAMPLES
    retrain_triggered = rate_ok and samples_ok

    if rate_ok and not samples_ok:
        logger.info(
            "drift_monitor: rate threshold met (%.4f) but only %d/%d confirmed samples — retrain deferred",
            rolling, confirmed_total, MIN_CONFIRMED_SAMPLES,
        )

    client = get_supabase_service_client()
    if client:
        try:
            client.table("drift_log").upsert(
                {
                    "date":              date_str,
                    "total_auto":        daily["total_auto"],
                    "disagreements":     daily["disagreements"],
                    "disagreement_rate": daily["disagreement_rate"],
                    "rolling_7d_rate":   rolling,
                    "retrain_triggered": retrain_triggered,
                    "computed_at":       datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="date",
            ).execute()
        except Exception as exc:
            logger.warning("drift_monitor: drift_log upsert failed: %s", exc)

    result = {
        **daily,
        "rolling_7d_rate":   rolling,
        "drift_threshold":   DRIFT_THRESHOLD,
        "retrain_triggered": retrain_triggered,
    }
    logger.info("drift_monitor: %s", result)
    return result
