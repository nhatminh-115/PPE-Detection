"""
Label persistence and crop upload helpers for Phase 1 (Human-in-the-Loop).

Crop normalization pipeline:
  - min side < SR_THRESHOLD px → SR upscaler (Real-ESRGAN x4 if model file present,
    otherwise bicubic 4x proxy), then INTER_AREA to 224x224.
  - otherwise → direct INTER_AREA resize to 224x224.
  fallback_used=True is persisted on the label row when the bicubic path is taken.

Hash-based dedupe:
  SHA-256 of the normalized JPEG bytes is used as the Storage object key
  (crops/by-hash/<sha256>.jpg). Existing objects are reused without re-upload.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

# cv2 and numpy are lazy-imported inside functions that need them (not available in CI
# environments that only run report/drift pipelines without OpenCV installed).

# Prevent torch from loading CUDA DLLs at import time (WinError 1455 on small paging files).
# Real-ESRGAN inference for crop normalization runs on CPU only — GPU is not needed here.
import os as _os
import sys as _sys
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# basicsr 1.4.2 references torchvision.transforms.functional_tensor which was
# removed in torchvision >= 0.17. Shim it before any realesrgan/basicsr import.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        import torchvision.transforms.functional as _ftf
        _sys.modules["torchvision.transforms.functional_tensor"] = _ftf
    except ImportError:
        pass

from src.infrastructure.supabase import get_supabase_service_client
from src.config import IMG_SIZE

logger = logging.getLogger(__name__)

CROPS_BUCKET  = "ppe-crops"
SR_THRESHOLD  = 16    # min side below this triggers SR upscaling first
ALL_PPE_ITEMS = ("hardhat", "vest")

# Supabase PostgREST error code when a table is missing from the schema cache.
_PGRST_TABLE_NOT_FOUND = "PGRST205"
_migration_warned = False


def _is_table_missing(exc: Exception) -> bool:
    msg = str(exc)
    return _PGRST_TABLE_NOT_FOUND in msg or ("ppe_labels" in msg and "schema cache" in msg)


def _warn_migration_once() -> None:
    global _migration_warned
    if not _migration_warned:
        logger.warning(
            "ppe_labels table not found. "
            "Run supabase/migrations/001_ppe_labels.sql in the Supabase SQL editor, "
            "then create a 'ppe-crops' Storage bucket (public). "
            "Labels will not be persisted until the migration is applied."
        )
        _migration_warned = True


# ── SR upscaler (Real-ESRGAN x4) ──────────────────────────────────────────────

_REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
_REALESRGAN_MODEL_FILE = "RealESRGAN_x4plus.pth"
_SR_SCALE = 4


class _SRUpscaler:
    """
    Real-ESRGAN x4 upscaler via the realesrgan package.

    Load order:
      1. SR_MODEL_PATH env var (explicit .pth override)
      2. model_cache/RealESRGAN_x4plus.pth (local cache)
      3. Auto-download RealESRGAN_x4plus.pth from xinntao GitHub releases

    Falls back to bicubic INTER_CUBIC when the model file is unavailable
    or the realesrgan package is not installed.
    """

    def __init__(self) -> None:
        self._upsampler = None
        self._ready = False
        self._try_load()

    def _try_load(self) -> None:
        import os
        os.makedirs("model_cache", exist_ok=True)

        # 1. Explicit override via env var
        override = os.environ.get("SR_MODEL_PATH", "")
        if override and os.path.isfile(override):
            if self._load_file(override):
                return

        # 2. Local cache
        cached_path = os.path.join("model_cache", _REALESRGAN_MODEL_FILE)
        if not os.path.isfile(cached_path):
            self._download(_REALESRGAN_MODEL_URL, cached_path)
        if os.path.isfile(cached_path) and self._load_file(cached_path):
            return

        logger.info(
            "Real-ESRGAN model unavailable — bicubic fallback active. "
            "Set SR_MODEL_PATH or ensure model_cache/ is writable for auto-download."
        )

    def _download(self, url: str, dest: str) -> None:
        import urllib.request
        try:
            logger.info("Downloading Real-ESRGAN model: %s → %s", url, dest)
            urllib.request.urlretrieve(url, dest)
            logger.info("Real-ESRGAN model downloaded: %s", dest)
        except Exception as exc:
            logger.warning("Real-ESRGAN model download failed (%s): %s", url, exc)
            import os
            if os.path.exists(dest):
                os.remove(dest)

    def _load_file(self, model_path: str) -> bool:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32,
                scale=_SR_SCALE,
            )
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except OSError:
                device = "cpu"
            self._upsampler = RealESRGANer(
                scale=_SR_SCALE,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device=device,
            )
            self._ready = True
            logger.info("Real-ESRGAN model loaded: %s", model_path)
            return True
        except ImportError as exc:
            logger.warning(
                "realesrgan/basicsr package not installed (%s) — bicubic fallback active. "
                "Run: pip install realesrgan",
                exc,
            )
            return False
        except OSError as exc:
            logger.warning(
                "torch failed to load (%s) — bicubic fallback active. "
                "Set CUDA_VISIBLE_DEVICES='' or increase Windows paging file size.",
                exc,
            )
            return False
        except Exception as exc:
            logger.debug("Real-ESRGAN model load failed (%s): %s", model_path, exc)
            return False

    def upscale(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Upscale image by 4x. Returns (upscaled_img, fallback_used).
        fallback_used=True means bicubic was used instead of Real-ESRGAN.
        """
        if self._ready and self._upsampler is not None:
            try:
                # RealESRGANer.enhance expects BGR numpy array, returns (output, _)
                output, _ = self._upsampler.enhance(img, outscale=_SR_SCALE)
                return output, False
            except Exception as exc:
                logger.debug("Real-ESRGAN upsample failed, falling back to bicubic: %s", exc)
        import cv2
        h, w = img.shape[:2]
        return cv2.resize(img, (w * _SR_SCALE, h * _SR_SCALE), interpolation=cv2.INTER_CUBIC), True


_upscaler: Optional[_SRUpscaler] = None


def _get_upscaler() -> _SRUpscaler:
    global _upscaler
    if _upscaler is None:
        _upscaler = _SRUpscaler()
    return _upscaler


# ── Crop normalization ─────────────────────────────────────────────────────────

def normalize_crop(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return (224x224 image, fallback_used).
    fallback_used=True when the SR model was unavailable and bicubic was used instead.
    """
    import cv2
    h, w = img.shape[:2]
    fallback_used = False
    if min(h, w) < SR_THRESHOLD:
        img, fallback_used = _get_upscaler().upscale(img)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), fallback_used


# ── Verdict derivation (authoritative — backend owns this) ─────────────────────

def derive_verdicts(predicted_missing: list[str], human_missing: list[str]) -> tuple[dict, str]:
    """
    Compute per-item TP/FP/FN/TN verdicts and aggregate badge from predicted and human labels.
    Returns (per_item dict, aggregate string).
    """
    per_item: dict[str, str] = {}
    for item in ALL_PPE_ITEMS:
        flagged  = item in predicted_missing
        confirmed = item in human_missing
        if flagged and confirmed:       per_item[item] = "tp"
        elif flagged and not confirmed: per_item[item] = "fp"
        elif not flagged and confirmed: per_item[item] = "fn"
        else:                           per_item[item] = "tn"

    verdicts = list(per_item.values())
    if any(v == "fp" for v in verdicts):   aggregate = "has-fp"
    elif any(v == "fn" for v in verdicts): aggregate = "has-fn"
    else:                                   aggregate = "all-tp"
    return per_item, aggregate


# ── Model metadata resolver ────────────────────────────────────────────────────

from src.config import MLFLOW_RUN_ID, EFFNET_RUN_ID

# Map reported_by_model string → (model_name, model_version, mlflow_run_id | None)
# run_id here is the MLflow training run that produced the model weights — NOT session_id.
# SigLIP/CLIP are zero-shot pretrained models with no MLflow run; run_id is None for them.
_MODEL_METADATA_MAP: dict[str, tuple[str, str, str | None]] = {
    "yolo":    ("YOLOv8",          "v1",      MLFLOW_RUN_ID),
    "effnet":  ("EfficientNet-B0", "stage2",  EFFNET_RUN_ID),
    "siglip":  ("SigLIP-SO400M",   "webli",   None),
    "clip":    ("SigLIP-SO400M",   "webli",   None),
    "unknown": ("unknown",         "",        None),
}


def _resolve_model_metadata(row_ids: list[str], fallback_model: str | None) -> dict:
    """
    Look up model metadata from source ppe_violations rows.
    Returns dict with model_name, model_version, run_id keys.
    """
    client = get_supabase_service_client()
    reporter = (fallback_model or "unknown").lower()

    if client and row_ids:
        try:
            res = (
                client.table("ppe_violations")
                .select("reported_by_model")
                .in_("id", row_ids[:5])   # sample first 5, all should agree
                .limit(5)
                .execute()
            )
            for row in (res.data or []):
                m = (row.get("reported_by_model") or "").lower()
                if m and m != "unknown":
                    reporter = m
                    break
        except Exception:
            pass

    # Match reporter string to metadata map (prefix match)
    for key, (name, version, run_id) in _MODEL_METADATA_MAP.items():
        if key in reporter:
            return {"model_name": name, "model_version": version, "run_id": run_id or None}

    return {"model_name": reporter or None, "model_version": None, "run_id": None}


# ── Crop upload with hash dedupe ───────────────────────────────────────────────

def upload_crop_to_storage(
    image_path: str,
    merge_key: str,
    img_array=None,
) -> tuple[str | None, str | None, bool]:
    """
    Normalize crop to 224x224, upload to Supabase Storage.
    Returns (public_url, sha256_hex, fallback_used).

    img_array: pass numpy BGR array directly to skip disk read.
    """
    client = get_supabase_service_client()
    if not client:
        return None, None, False

    try:
        import cv2
        if img_array is not None:
            img = img_array
        else:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning("Crop not found on disk: %s", image_path)
                return None, None, False

        img_224, fallback_used = normalize_crop(img)
        ok, buf = cv2.imencode(".jpg", img_224, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None, None, False

        data = buf.tobytes()
        sha256 = hashlib.sha256(data).hexdigest()

        # Dedupe: if another label already stored the same normalized image, reuse its URL
        try:
            res = (
                client.table("ppe_labels")
                .select("crop_url")
                .eq("crop_sha256", sha256)
                .limit(1)
                .execute()
            )
            if res.data and res.data[0].get("crop_url"):
                logger.debug("Crop reused (sha256 match): %s", sha256[:12])
                return res.data[0]["crop_url"], sha256, fallback_used
        except Exception:
            pass

        safe_key = merge_key.replace("/", "_").replace("\\", "_").replace(" ", "_")
        storage_path = f"crops/{safe_key}.jpg"

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                client.storage.from_(CROPS_BUCKET).upload(
                    storage_path,
                    data,
                    {"content-type": "image/jpeg", "upsert": "true"},
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    import time as _time
                    _time.sleep(0.5 * (attempt + 1))
        else:
            raise last_exc  # type: ignore[misc]

        public_url = client.storage.from_(CROPS_BUCKET).get_public_url(storage_path)
        logger.info("Crop uploaded: %s → sha256:%s", merge_key, sha256[:12])
        return public_url, sha256, fallback_used

    except Exception as exc:
        logger.warning("Crop upload failed for %s: %s", merge_key, exc)
        return None, None, False


# ── Label CRUD ────────────────────────────────────────────────────────────────

def upsert_label(payload: dict, labeled_by: str = "anonymous") -> dict:
    """
    Upsert a label into ppe_labels and append an audit record.

    Required payload keys: merge_key, session_id, tracker_id, predicted_missing,
      human_missing (or skipped=True), row_ids.
    per_item and aggregate are computed server-side and override any frontend values.
    """
    client = get_supabase_service_client()
    if not client:
        raise RuntimeError("Supabase unavailable")

    merge_key      = payload["merge_key"]
    skipped        = bool(payload.get("skipped", False))
    predicted      = [str(x) for x in (payload.get("predicted_missing") or [])]
    human          = [str(x) for x in (payload.get("human_missing") or [])] if not skipped else []

    # Backend owns verdict derivation
    if skipped:
        per_item, aggregate = {}, "skip"
    else:
        per_item, aggregate = derive_verdicts(predicted, human)

    # Resolve model metadata from source violations
    meta = _resolve_model_metadata(
        row_ids=payload.get("row_ids") or [],
        fallback_model=payload.get("model"),
    )

    # Fetch existing row for audit diff and crop reuse check
    existing = None
    try:
        res = client.table("ppe_labels").select("*").eq("merge_key", merge_key).limit(1).execute()
        existing = res.data[0] if res.data else None
    except Exception:
        pass

    row: dict = {
        "merge_key":          merge_key,
        "session_id":         str(payload.get("session_id", "")),
        "tracker_id":         str(payload.get("tracker_id", "")),
        "predicted_missing":  predicted,
        "human_missing":      human if not skipped else None,
        "per_item":           per_item,
        "aggregate":          aggregate,
        "skipped":            skipped,
        "model":              payload.get("model"),
        "row_ids":            payload.get("row_ids") or [],
        "first_created_at":   payload.get("first_created_at"),
        "labeled_by":         labeled_by,
        "labeled_at":         datetime.now(timezone.utc).isoformat(),
        "model_name":         meta["model_name"],
        "model_version":      meta["model_version"],
        "run_id":             meta["run_id"],
        "threshold_snapshot": payload.get("threshold_snapshot") or {},
        "is_auto_labeled":    bool(payload.get("is_auto_labeled", False)),
        "auto_label_model":   payload.get("auto_label_model"),
        "expire_at":          (datetime.now(timezone.utc) + timedelta(days=2 if payload.get("is_auto_labeled") else 3)).isoformat(),
    }

    # Crop upload — skip if already uploaded for this merge_key
    image_path = payload.get("image_path")
    if image_path and (not existing or not existing.get("crop_url")):
        url, sha256, fallback_used = upload_crop_to_storage(image_path, merge_key)
        if url:
            row["crop_url"]          = url
            row["crop_uploaded_at"]  = datetime.now(timezone.utc).isoformat()
            row["crop_sha256"]       = sha256
            row["fallback_used"]     = fallback_used

    try:
        upserted = (
            client.table("ppe_labels")
            .upsert(row, on_conflict="merge_key")
            .execute()
        )
    except Exception as exc:
        if _is_table_missing(exc):
            _warn_migration_once()
            raise RuntimeError("ppe_labels table missing — run migration first") from exc
        raise

    saved_row = upserted.data[0] if upserted.data else row

    # Audit log
    action = "update" if existing else "create"
    if skipped and existing and not existing.get("skipped"):
        action = "skip"
    elif not skipped and existing and existing.get("skipped"):
        action = "unskip"

    try:
        client.table("ppe_label_audit").insert({
            "label_id":       saved_row.get("id"),
            "merge_key":      merge_key,
            "action":         action,
            "previous_state": existing,
            "new_state":      saved_row,
            "labeled_by":     labeled_by,
        }).execute()
    except Exception as exc:
        if not _is_table_missing(exc):
            logger.warning("Audit insert failed for %s: %s", merge_key, exc)

    return saved_row


def list_labels(
    aggregate: str | None = None,
    session_id: str | None = None,
    limit: int = 500,
) -> list[dict]:
    client = get_supabase_service_client()
    if not client:
        return []
    try:
        q = client.table("ppe_labels").select("*")
        if aggregate:
            q = q.eq("aggregate", aggregate)
        if session_id:
            q = q.eq("session_id", session_id)
        return q.order("labeled_at", desc=True).limit(min(limit, 2000)).execute().data or []
    except Exception as exc:
        if _is_table_missing(exc):
            _warn_migration_once()
        else:
            logger.warning("list_labels failed: %s", exc)
        return []


def backfill_model_metadata() -> dict:
    """
    One-time backfill: set model_name/model_version/run_id on existing ppe_labels rows
    where model_name is null. Resolves from ppe_violations via row_ids.
    Returns a summary dict.
    """
    client = get_supabase_service_client()
    if not client:
        return {"updated": 0, "skipped": 0, "error": "Supabase unavailable"}

    try:
        rows = (
            client.table("ppe_labels")
            .select("id, merge_key, row_ids, model")
            .is_("model_name", "null")
            .limit(500)
            .execute()
            .data or []
        )
    except Exception as exc:
        return {"updated": 0, "skipped": 0, "error": str(exc)}

    updated = skipped = 0
    for row in rows:
        meta = _resolve_model_metadata(row.get("row_ids") or [], row.get("model"))
        if not meta["model_name"] or meta["model_name"] == "unknown":
            skipped += 1
            continue
        try:
            client.table("ppe_labels").update(meta).eq("id", row["id"]).execute()
            updated += 1
        except Exception:
            skipped += 1

    return {"updated": updated, "skipped": skipped, "total_processed": len(rows)}


def get_daily_quality(date_str: str | None = None) -> dict:
    """
    Item-level TP/FP/FN/TN and precision/recall for a given ICT calendar day.
    date_str: 'YYYY-MM-DD'. Defaults to today ICT.
    """
    ICT = timezone(timedelta(hours=7))
    if date_str:
        try:
            day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ICT)
        except ValueError:
            day = datetime.now(ICT)
    else:
        day = datetime.now(ICT)

    day_start = day.replace(hour=0,  minute=0,  second=0,  microsecond=0)
    day_end   = day.replace(hour=23, minute=59, second=59, microsecond=999999)

    client = get_supabase_service_client()
    if not client:
        return _empty_quality(date_str)

    try:
        rows = (
            client.table("ppe_labels")
            .select("per_item, skipped, aggregate")
            .gte("labeled_at", day_start.isoformat())
            .lte("labeled_at", day_end.isoformat())
            .eq("is_auto_labeled", False)   # human labels only for quality metrics
            .execute()
            .data or []
        )
    except Exception as exc:
        if _is_table_missing(exc):
            _warn_migration_once()
        else:
            logger.warning("daily_quality query failed: %s", exc)
        return _empty_quality(date_str)

    tp = fp = fn = tn = skip = 0
    for row in rows:
        if row.get("skipped"):
            skip += 1
            continue
        for verdict in (row.get("per_item") or {}).values():
            if verdict == "tp":   tp += 1
            elif verdict == "fp": fp += 1
            elif verdict == "fn": fn += 1
            elif verdict == "tn": tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None

    return {
        "date":                date_str or day.strftime("%Y-%m-%d"),
        "labeled_events":      len(rows),
        "skipped":             skip,
        "item_tp":             tp,
        "item_fp":             fp,
        "item_fn":             fn,
        "item_tn":             tn,
        "precision":           round(precision, 4) if precision is not None else None,
        "recall":              round(recall, 4)    if recall    is not None else None,
        "label_coverage_note": "sparse" if len(rows) < 10 else "adequate",
    }


def _empty_quality(date_str: str | None) -> dict:
    return {
        "date": date_str or "",
        "labeled_events": 0, "skipped": 0,
        "item_tp": 0, "item_fp": 0, "item_fn": 0, "item_tn": 0,
        "precision": None, "recall": None,
        "label_coverage_note": "no_data",
    }


# ── Crop retention cleanup ─────────────────────────────────────────────────────

def cleanup_expired_crops() -> dict:
    """
    Purge expired crop files from Supabase Storage and null out stale crop refs in ppe_labels.

    Retention policy (set at label creation):
      - Scout auto-labels (is_auto_labeled=True):  expire_at = labeled_at + 2 days
      - Human labels     (is_auto_labeled=False):  expire_at = labeled_at + 3 days

    Safety guarantees:
      - A storage object is only deleted when no non-expired label shares the same crop_sha256.
      - DB crop refs are only nulled after the storage object is successfully removed (or when
        the object is shared and we skip deletion — to keep the expired row clean).
      - If storage removal fails, DB refs are left intact so the next run can retry.

    Returns a summary dict with keys: removed, nulled, skipped_shared, total_expired.
    """
    client = get_supabase_service_client()
    if not client:
        return {"removed": 0, "nulled": 0, "skipped_shared": 0, "total_expired": 0, "error": "Supabase unavailable"}

    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()

    def _is_active_ref(expire_at_raw: str | None) -> bool:
        # Treat null/invalid expire_at as active to avoid accidental deletion.
        if not expire_at_raw:
            return True
        try:
            expire_dt = datetime.fromisoformat(str(expire_at_raw).replace("Z", "+00:00"))
            return expire_dt > now_dt
        except Exception:
            return True

    try:
        rows = (
            client.table("ppe_labels")
            .select("id, merge_key, crop_url, crop_sha256")
            .or_(f"expire_at.lt.{now_iso},expire_at.is.null")
            .not_.is_("crop_url", "null")
            .execute()
            .data or []
        )
    except Exception as exc:
        logger.warning("cleanup_expired_crops: query failed: %s", exc)
        return {"removed": 0, "nulled": 0, "skipped_shared": 0, "total_expired": 0, "error": str(exc)}

    removed = nulled = skipped_shared = 0

    for row in rows:
        crop_url    = row.get("crop_url") or ""
        crop_sha256 = row.get("crop_sha256")
        row_id      = row["id"]
        merge_key   = row.get("merge_key", "?")

        has_storage_path = f"/{CROPS_BUCKET}/" in crop_url
        storage_deleted  = False
        shared_object    = False

        if has_storage_path:
            path_part = crop_url.split(f"/{CROPS_BUCKET}/", 1)[-1].split("?")[0]
            if path_part == crop_url:
                logger.warning("cleanup: unexpected crop_url format, skipping: %s", crop_url[:80])
                continue

            # Safety check: skip deletion if any non-expired label still references
            # the same hash (deduped objects are shared across rows).
            if crop_sha256:
                try:
                    refs = (
                        client.table("ppe_labels")
                        .select("id, expire_at")
                        .eq("crop_sha256", crop_sha256)
                        .neq("id", row_id)
                        .not_.is_("crop_url", "null")
                        .execute()
                    )
                    if any(_is_active_ref(r.get("expire_at")) for r in (refs.data or [])):
                        shared_object = True
                except Exception as exc:
                    # On query failure, treat conservatively: assume active ref exists.
                    logger.debug(
                        "cleanup_expired_crops: active-ref check failed for %s: %s",
                        merge_key, exc,
                    )
                    shared_object = True

            # Fallback safety check by exact crop_url path when sha is missing or migrated data is inconsistent.
            if not shared_object:
                try:
                    refs_by_url = (
                        client.table("ppe_labels")
                        .select("id, expire_at, crop_url")
                        .eq("crop_url", crop_url)
                        .neq("id", row_id)
                        .execute()
                    )
                    if any(_is_active_ref(r.get("expire_at")) for r in (refs_by_url.data or [])):
                        shared_object = True
                except Exception as exc:
                    logger.debug(
                        "cleanup_expired_crops: active-ref(url) check failed for %s: %s",
                        merge_key, exc,
                    )
                    shared_object = True

            if shared_object:
                # Object still needed by another label — skip storage delete.
                # Still null this expired row's pointer so it stops appearing in cleanup queries.
                skipped_shared += 1
                try:
                    client.table("ppe_labels").update({
                        "crop_url":         None,
                        "crop_sha256":      None,
                        "crop_uploaded_at": None,
                    }).eq("id", row_id).execute()
                    nulled += 1
                except Exception as exc:
                    logger.debug(
                        "cleanup_expired_crops: null crop ref (shared) failed for %s: %s",
                        merge_key, exc,
                    )
                continue

            # No active reference — safe to remove from Storage.
            try:
                client.storage.from_(CROPS_BUCKET).remove([path_part])
                removed += 1
                storage_deleted = True
            except Exception as exc:
                logger.debug(
                    "cleanup_expired_crops: storage remove failed for %s: %s",
                    merge_key, exc,
                )

        # Null DB crop refs only when:
        #   (a) storage was successfully removed, or
        #   (b) this row had no storage path to begin with (crop_url points elsewhere).
        if storage_deleted or not has_storage_path:
            try:
                client.table("ppe_labels").update({
                    "crop_url":         None,
                    "crop_sha256":      None,
                    "crop_uploaded_at": None,
                }).eq("id", row_id).execute()
                nulled += 1
            except Exception as exc:
                logger.debug(
                    "cleanup_expired_crops: null crop ref failed for %s: %s",
                    merge_key, exc,
                )

    logger.info(
        "cleanup_expired_crops: removed=%d objects, nulled=%d rows, "
        "skipped_shared=%d, total_expired=%d",
        removed, nulled, skipped_shared, len(rows),
    )
    return {
        "removed":        removed,
        "nulled":         nulled,
        "skipped_shared": skipped_shared,
        "total_expired":  len(rows),
    }
