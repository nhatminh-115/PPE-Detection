"""
Phase 3 — Second Opinion Agent

End-of-day batch job that:
  1. Finds uncertain violation events (avg confidence < UNCERTAIN_THRESHOLD).
  2. Sends each crop to Llama 4 Scout (Groq vision API) for a second opinion.
  3. Stores the result in ppe_labels with is_auto_labeled=True.
  4. Disagreements (model flagged violation, Scout disagrees) become priority
     items in the human review queue.

Auto-labels are excluded from precision/recall in daily quality reports.
They are used only to prioritise which events humans should review first.

Entry point:
    POST /api/second_opinion/run?date=YYYY-MM-DD
    or:
    from src.api.second_opinion import run_second_opinion_batch
    result = await run_second_opinion_batch(date_str)
"""

from __future__ import annotations

import base64
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

ICT = timezone(timedelta(hours=7))

# Confidence below this triggers second opinion.
UNCERTAIN_THRESHOLD = float(os.environ.get("SECOND_OPINION_THRESHOLD", "1.0"))

# Groq model for vision second opinion.
SCOUT_MODEL = os.environ.get("SECOND_OPINION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

PPE_ITEMS = ("hardhat", "vest")


# ── Utility ───────────────────────────────────────────────────────────────────

def _encode_image(path: str) -> str | None:
    """Return base64-encoded JPEG string, or None if file not found."""
    clean = path.lstrip("/")
    if not os.path.isfile(clean):
        logger.debug("second_opinion: image not found: %s", clean)
        return None
    try:
        with open(clean, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logger.debug("second_opinion: image read failed (%s): %s", clean, exc)
        return None


def _avg_conf(violations: list[dict]) -> float | None:
    confs = [v.get("confidence") for v in violations if v.get("confidence") is not None]
    return sum(confs) / len(confs) if confs else None


# ── Step 1: find uncertain events for a date ──────────────────────────────────

def find_uncertain_events(date_str: str) -> list[dict]:
    """
    Return merged (session_id, tracker_id) events from ppe_violations where
    avg confidence < UNCERTAIN_THRESHOLD for the given ICT date.

    Returns list of dicts:
      { merge_key, session_id, tracker_id, row_ids, predicted_missing,
        avg_conf, image_path, model, first_created_at }
    """
    from src.infrastructure.supabase import get_supabase_service_client

    client = get_supabase_service_client()
    if not client:
        return []

    day_start = f"{date_str}T00:00:00+07:00"
    day_end   = f"{date_str}T23:59:59+07:00"

    try:
        rows = (
            client.table("ppe_violations")
            .select("*")
            .gte("created_at", day_start)
            .lt("created_at", day_end)
            .execute()
            .data or []
        )
    except Exception as exc:
        logger.warning("second_opinion: violations query failed: %s", exc)
        return []

    # Merge by (session_id, tracker_id) — same logic as Label Studio UI
    merged: dict[str, dict] = {}
    for row in rows:
        sid = str(row.get("session_id") or "nosession")
        viols = row.get("violations") or []
        tid = str(viols[0].get("tracker_id", "?")) if viols else "?"
        key = f"{sid}__{tid}"

        if key not in merged:
            merged[key] = {
                "merge_key":       key,
                "session_id":      sid,
                "tracker_id":      tid,
                "row_ids":         [],
                "predicted_missing": [],
                "conf_sum":        0.0,
                "conf_n":          0,
                "image_path":      None,
                "model":           row.get("reported_by_model") or "unknown",
                "first_created_at": row.get("created_at"),
            }

        ev = merged[key]
        ev["row_ids"].append(str(row["id"]))

        for v in viols:
            vtype = v.get("violation_type", "")
            item = vtype.replace("none_", "")
            if item and item not in ev["predicted_missing"]:
                ev["predicted_missing"].append(item)
            conf = v.get("confidence")
            if conf is not None:
                ev["conf_sum"] += float(conf)
                ev["conf_n"]   += 1

        if not ev["image_path"]:
            vp = viols[0].get("image_path") if viols else None
            if vp:
                ev["image_path"] = vp

        if row.get("created_at", "") < ev["first_created_at"]:
            ev["first_created_at"] = row["created_at"]

    # Filter to uncertain events only
    uncertain = []
    for ev in merged.values():
        avg = ev["conf_sum"] / ev["conf_n"] if ev["conf_n"] > 0 else None
        ev["avg_conf"] = avg
        del ev["conf_sum"], ev["conf_n"]
        if avg is None or avg < UNCERTAIN_THRESHOLD:
            uncertain.append(ev)

    logger.info(
        "second_opinion: %d total events, %d uncertain (conf < %.2f) on %s",
        len(merged), len(uncertain), UNCERTAIN_THRESHOLD, date_str,
    )
    return uncertain


# ── Step 2: call Llama 4 Scout on a single crop ───────────────────────────────

def _build_vision_prompt(predicted_missing: list[str]) -> str:
    items = " and ".join(predicted_missing) if predicted_missing else "PPE items"
    return (
        f"This is a cropped image of a person from a construction site safety camera.\n"
        f"The automated detection system flagged this person as missing: {items}.\n\n"
        f"Look carefully at the image and answer for each item:\n"
        + "\n".join(f"- Is the person missing a {item}? Answer YES or NO." for item in (predicted_missing or PPE_ITEMS))
        + "\n\nRespond with a JSON object only, e.g.: "
        + '{"hardhat": "YES", "vest": "NO"}'
        + "\nUse only the item names listed above as keys."
    )


def call_scout(image_path: str, predicted_missing: list[str]) -> dict | None:
    """
    Send crop to Llama 4 Scout via Groq vision API.
    Returns dict like {"hardhat": True, "vest": False} where True = missing.
    Returns None if call fails or image unavailable.
    """
    import json
    from groq import Groq

    b64 = _encode_image(image_path)
    if not b64:
        return None

    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=SCOUT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":  "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": _build_vision_prompt(predicted_missing),
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=128,
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON from response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logger.debug("second_opinion: Scout response has no JSON: %s", raw[:200])
            return None

        parsed = json.loads(raw[start:end])
        # Normalise to bool: "YES" / "yes" / True → True
        return {
            k.lower(): str(v).strip().upper() == "YES"
            for k, v in parsed.items()
        }

    except Exception as exc:
        logger.warning("second_opinion: Scout call failed: %s", exc)
        return None


# ── Step 3: store auto-label ──────────────────────────────────────────────────

def store_auto_label(ev: dict, scout_result: dict) -> bool:
    """
    Write a ppe_labels row for an auto-labeled event.
    scout_result: {item: True/False} where True = Scout thinks item IS missing.
    Returns True on success.
    """
    from src.api.label_service import (
        derive_verdicts,
        upload_crop_to_storage,
        _resolve_model_metadata,
    )
    from src.infrastructure.supabase import get_supabase_service_client

    client = get_supabase_service_client()
    if not client:
        return False

    human_missing = [item for item, missing in scout_result.items() if missing]
    predicted_missing = ev.get("predicted_missing") or []

    per_item, aggregate = derive_verdicts(
        predicted_missing=["none_" + i for i in predicted_missing],
        human_missing=["none_" + i for i in human_missing],
    )

    has_disagreement = any(v in ("fp", "fn") for v in per_item.values())

    # Resolve model metadata from source violations
    meta = _resolve_model_metadata(
        row_ids=ev.get("row_ids") or [],
        fallback_model=ev.get("model"),
    )

    row = {
        "merge_key":        ev["merge_key"],
        "session_id":       ev["session_id"],
        "tracker_id":       ev["tracker_id"],
        "predicted_missing": predicted_missing,
        "human_missing":    human_missing,
        "per_item":         per_item,
        "aggregate":        aggregate,
        "skipped":          False,
        "model":            ev.get("model"),
        "row_ids":          ev["row_ids"],
        "first_created_at": ev.get("first_created_at"),
        "labeled_by":       "llama4-scout",
        "labeled_at":       datetime.now(timezone.utc).isoformat(),
        "is_auto_labeled":  True,
        "auto_label_model": SCOUT_MODEL,
        "model_name":       meta["model_name"],
        "model_version":    meta["model_version"],
        "run_id":           meta["run_id"],
        "threshold_snapshot": {
            "avg_conf":            ev.get("avg_conf"),
            "uncertain_threshold": UNCERTAIN_THRESHOLD,
        },
    }

    # Upload normalized crop — same deduplication path as human labels
    image_path = ev.get("image_path")
    if image_path:
        url, sha256, fallback_used = upload_crop_to_storage(image_path, ev["merge_key"])
        if url:
            row["crop_url"]         = url
            row["crop_uploaded_at"] = datetime.now(timezone.utc).isoformat()
            row["crop_sha256"]      = sha256
            row["fallback_used"]    = fallback_used

    try:
        client.table("ppe_labels").upsert(row, on_conflict="merge_key").execute()
        logger.debug(
            "second_opinion: stored auto-label for %s (disagreement=%s)",
            ev["merge_key"], has_disagreement,
        )
        return True
    except Exception as exc:
        logger.warning("second_opinion: store failed for %s: %s", ev["merge_key"], exc)
        return False


# ── Batch entrypoint ──────────────────────────────────────────────────────────

async def run_second_opinion_batch(date_str: str | None = None) -> dict:
    """
    Run the full second opinion batch for a given ICT date.
    Skips events already labeled by a human (is_auto_labeled=False rows take precedence).

    Returns a summary dict.
    """
    import asyncio
    from src.infrastructure.supabase import get_supabase_service_client

    if date_str is None:
        date_str = datetime.now(ICT).strftime("%Y-%m-%d")

    uncertain = find_uncertain_events(date_str)
    if not uncertain:
        return {"date": date_str, "uncertain": 0, "processed": 0, "stored": 0, "skipped_human": 0, "errors": 0}

    # Skip events already labeled by a human
    client = get_supabase_service_client()
    human_labeled: set[str] = set()
    if client:
        try:
            keys = [ev["merge_key"] for ev in uncertain]
            res = (
                client.table("ppe_labels")
                .select("merge_key")
                .in_("merge_key", keys)
                .eq("is_auto_labeled", False)
                .execute()
            )
            human_labeled = {r["merge_key"] for r in (res.data or [])}
        except Exception as exc:
            logger.warning("second_opinion: human-label check failed: %s", exc)

    to_process = [ev for ev in uncertain if ev["merge_key"] not in human_labeled]
    skipped_human = len(uncertain) - len(to_process)

    logger.info(
        "second_opinion: %d uncertain, %d skipped (human-labeled), %d to process",
        len(uncertain), skipped_human, len(to_process),
    )

    processed = stored = errors = 0

    def _process_one(ev: dict) -> bool:
        if not ev.get("image_path"):
            return False
        result = call_scout(ev["image_path"], ev["predicted_missing"])
        if result is None:
            return False
        return store_auto_label(ev, result)

    loop = asyncio.get_running_loop()
    for ev in to_process:
        processed += 1
        try:
            ok = await loop.run_in_executor(None, _process_one, ev)
            if ok:
                stored += 1
            else:
                errors += 1
        except Exception as exc:
            logger.warning("second_opinion: event %s failed: %s", ev["merge_key"], exc)
            errors += 1

    summary = {
        "date":           date_str,
        "uncertain":      len(uncertain),
        "skipped_human":  skipped_human,
        "processed":      processed,
        "stored":         stored,
        "errors":         errors,
        "threshold":      UNCERTAIN_THRESHOLD,
        "model":          SCOUT_MODEL,
    }
    logger.info("second_opinion: batch complete: %s", summary)
    return summary
