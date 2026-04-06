"""
Phase 4 — S3 Dataset Uploader

Exports confirmed human labels (is_auto_labeled=False, skipped=False) plus their
crops from Supabase Storage to S3 in a versioned partition schema.

S3 layout:
  {prefix}/crops/dt={YYYY-MM-DD}/{sha256}.jpg   -- deduplicated by sha256
  {prefix}/labels/dt={YYYY-MM-DD}/labels.jsonl  -- label manifest, one JSON per line

Environment variables:
  S3_BUCKET            -- required
  S3_PREFIX            -- optional, default "ppe-flywheel"
  AWS_ACCESS_KEY_ID    -- required (or IAM role)
  AWS_SECRET_ACCESS_KEY -- required (or IAM role)
  AWS_DEFAULT_REGION   -- optional, default "us-east-1"
  SUPABASE_URL, SUPABASE_SERVICE_KEY
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timezone, timedelta

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "ppe-flywheel")
ICT = timezone(timedelta(hours=7))


def _s3_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _download_crop(crop_url: str) -> bytes | None:
    import requests as req
    try:
        resp = req.get(crop_url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("s3_uploader: crop download failed (%s): %s", crop_url, exc)
        return None


def _s3_object_exists(s3, key: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return False
        raise


def upload_date_to_s3(date_str: str) -> dict:
    """
    Export all confirmed labels for a given ICT date to S3.

    Crops are deduplicated by sha256 — re-uploading the same sha256 is skipped.
    Returns a summary dict.
    """
    from src.infrastructure.supabase import get_supabase_service_client

    if not S3_BUCKET:
        raise ValueError("S3_BUCKET env var is not set")

    client = get_supabase_service_client()
    if not client:
        raise RuntimeError("Supabase service client unavailable")

    day_start = f"{date_str}T00:00:00+07:00"
    day_end   = f"{date_str}T23:59:59+07:00"

    rows = (
        client.table("ppe_labels")
        .select("*")
        .eq("is_auto_labeled", False)
        .eq("skipped", False)
        .gte("labeled_at", day_start)
        .lt("labeled_at", day_end)
        .execute()
        .data or []
    )

    if not rows:
        logger.info("s3_uploader: no confirmed labels for %s", date_str)
        return {"date": date_str, "labels": 0, "crops_uploaded": 0, "crops_skipped": 0}

    s3             = _s3_client()
    crop_prefix    = f"{S3_PREFIX}/crops/dt={date_str}"
    labels_key     = f"{S3_PREFIX}/labels/dt={date_str}/labels.jsonl"
    crops_uploaded = 0
    crops_skipped  = 0
    manifest_lines: list[str] = []

    for row in rows:
        sha256   = row.get("crop_sha256")
        crop_url = row.get("crop_url")
        crop_key = f"{crop_prefix}/{sha256}.jpg" if sha256 else None

        if crop_key and crop_url:
            if _s3_object_exists(s3, crop_key):
                crops_skipped += 1
            else:
                image_bytes = _download_crop(crop_url)
                if image_bytes:
                    s3.put_object(
                        Bucket=S3_BUCKET,
                        Key=crop_key,
                        Body=image_bytes,
                        ContentType="image/jpeg",
                    )
                    crops_uploaded += 1
                    logger.debug("s3_uploader: uploaded %s", crop_key)
                else:
                    crops_skipped += 1
        else:
            crops_skipped += 1

        manifest_lines.append(json.dumps({
            "merge_key":         row.get("merge_key"),
            "session_id":        row.get("session_id"),
            "tracker_id":        row.get("tracker_id"),
            "predicted_missing": row.get("predicted_missing"),
            "human_missing":     row.get("human_missing"),
            "per_item":          row.get("per_item"),
            "aggregate":         row.get("aggregate"),
            "model_name":        row.get("model_name"),
            "model_version":     row.get("model_version"),
            "run_id":            row.get("run_id"),
            "crop_s3_key":       crop_key,
            "crop_sha256":       sha256,
            "labeled_at":        row.get("labeled_at"),
            "first_created_at":  row.get("first_created_at"),
        }))

    manifest_body = "\n".join(manifest_lines) + "\n"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=labels_key,
        Body=manifest_body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )

    result = {
        "date":           date_str,
        "labels":         len(rows),
        "crops_uploaded": crops_uploaded,
        "crops_skipped":  crops_skipped,
        "labels_key":     labels_key,
    }
    logger.info("s3_uploader: %s", result)
    return result
