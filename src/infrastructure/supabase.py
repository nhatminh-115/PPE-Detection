import os
import time
import logging
import concurrent.futures
import re
from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

supabase_client         = None
supabase_service_client = None
db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
reported_ids = set()


def get_supabase_client():
    """Anon-key client — used for reads and violation logging."""
    global supabase_client
    if supabase_client is None:
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Supabase connection successfully initialized.")
            except Exception as e:
                if "unexpected keyword argument 'proxy'" in str(e):
                    logger.error(
                        "Failed to initialize Supabase due to dependency mismatch. "
                        "Pin httpx to a compatible version (for example: httpx==0.25.2) "
                        "and reinstall requirements."
                    )
                logger.error(f"Failed to initialize Supabase: {e}")
        else:
            logger.warning("Supabase credentials missing. Remote logging disabled.")
    return supabase_client


def get_supabase_service_client():
    """
    Service-role client — bypasses RLS for backend writes (labels, audit, storage upload).
    Falls back to the anon client if SUPABASE_SERVICE_KEY is not set.
    Set SUPABASE_SERVICE_KEY in .env to enable full write access.
    """
    global supabase_service_client
    if supabase_service_client is None:
        key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
        if SUPABASE_URL and key:
            try:
                supabase_service_client = create_client(SUPABASE_URL, key)
                if SUPABASE_SERVICE_KEY:
                    logger.info("Supabase service-role client initialized.")
                else:
                    logger.warning(
                        "SUPABASE_SERVICE_KEY not set — label writes will use the anon key "
                        "and may be blocked by RLS. Add SUPABASE_SERVICE_KEY to .env."
                    )
            except Exception as e:
                logger.error(f"Failed to initialize Supabase service client: {e}")
        else:
            logger.warning("Supabase credentials missing. Label writes disabled.")
    return supabase_service_client


def send_telegram_alert(tracker_id: int, missing_items: list, crop_img) -> None:
    """Sends instant Telegram alert with violation crop image."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    logger.info(f"Telegram alert: token={'SET' if token else 'MISSING'}, chat_id={'SET' if chat_id else 'MISSING'}")

    if not token or not chat_id:
        return

    missing_str = ", ".join(item.upper() for item in missing_items)
    caption = (
        f"🚨 PPE Violation Detected\n"
        f"Tracker ID : {tracker_id}\n"
        f"Missing    : {missing_str}\n"
        f"Time       : {time.strftime('%H:%M:%S %d/%m/%Y')}"
    )

    try:
        import requests as _requests
        if crop_img is not None and crop_img.size > 0:
            # Send image with caption
            import cv2
            ret, buffer = cv2.imencode(".jpg", crop_img)
            if ret:
                _requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": ("violation.jpg", buffer.tobytes(), "image/jpeg")},
                    timeout=10,
                )
                return

        # Fallback: text only if no image
        _requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": caption},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")


def log_violation_to_supabase(tracker_id, missing_items, missing_probs, crop_img, session_id=None, reporter_model=None):

    """Asynchronously dispatches violation events and localized crops to Supabase."""
    try:
        timestamp_now = int(time.time())
        img_filename = f"violation_crops/ID{tracker_id}_{timestamp_now}.jpg"

        if crop_img is not None and crop_img.size > 0:
            import cv2
            import os
            os.makedirs("violation_crops", exist_ok=True)
            cv2.imwrite(img_filename, crop_img)

        # Upload crop to Supabase Storage so second_opinion can resolve the image
        # without depending on a shared local disk path.
        crop_url: str | None = None
        if crop_img is not None and crop_img.size > 0:
            try:
                from src.api.label_service import upload_crop_to_storage
                merge_key = f"violations/ID{tracker_id}_{timestamp_now}"
                url, _sha256, _fallback = upload_crop_to_storage(img_filename, merge_key)
                crop_url = url
            except Exception as _upload_exc:
                logger.debug("violation crop cloud upload skipped: %s", _upload_exc)

        violations_list = [
            {
                "tracker_id": int(tracker_id),
                "image_path": img_filename,
                "crop_url":   crop_url,   # None when upload failed; second_opinion falls back to image_path
                "violation_type": f"none_{item}",
                "confidence": float(prob),
                "reported_by_model": str(reporter_model or "unknown").lower(),
            }
            for item, prob in zip(missing_items, missing_probs)
        ]

        data = {
            "violations": violations_list,
            "status": "Warning",
            "reported_by_model": str(reporter_model or "unknown").lower(),
        }
        if session_id is not None:
            data["session_id"] = session_id

        client = get_supabase_client()
        if client:
            insert_payload = dict(data)
            for _ in range(2):
                try:
                    client.table("ppe_violations").insert(insert_payload).execute()
                    break
                except Exception as db_ex:
                    msg = str(db_ex)
                    m = re.search(r"Could not find the '([^']+)' column", msg)
                    if not m:
                        raise
                    missing_col = m.group(1)
                    if missing_col not in insert_payload:
                        raise
                    # Drop unknown top-level column and retry once.
                    insert_payload.pop(missing_col, None)
            else:
                raise RuntimeError("Failed to insert ppe_violations after schema fallback retries")
            logger.info(f"Database sync successful for violation ID: {tracker_id}.")

        # Instant Telegram alert with crop image
        send_telegram_alert(tracker_id, missing_items, crop_img)

    except Exception as e:
        logger.error(f"Database logging execution failed: {e}")