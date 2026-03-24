import os
import time
import logging
import concurrent.futures
import cv2
import requests
from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

supabase_client = None
db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
reported_ids = set()

def get_supabase_client():
    """Implements lazy initialization for Supabase to prevent CI/CD import crashes."""
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
                        "Pin httpx to a compatible version (for example: httpx==0.27.2) "
                        "and reinstall requirements."
                    )
                logger.error(f"Failed to initialize Supabase: {e}")
        else:
            logger.warning("Supabase credentials missing. Remote logging disabled.")
    return supabase_client


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
        if crop_img is not None and crop_img.size > 0:
            # Send image with caption
            ret, buffer = cv2.imencode(".jpg", crop_img)
            if ret:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": ("violation.jpg", buffer.tobytes(), "image/jpeg")},
                    timeout=10,
                )
                return

        # Fallback: text only if no image
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": caption},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")


def log_violation_to_supabase(tracker_id, missing_items, missing_probs, crop_img, session_id=None):

    """Asynchronously dispatches violation events and localized crops to Supabase."""
    try:
        timestamp_now = int(time.time())
        img_filename = f"violation_crops/ID{tracker_id}_{timestamp_now}.jpg"

        if crop_img is not None and crop_img.size > 0:
            cv2.imwrite(img_filename, crop_img)

        violations_list = [
            {
                "tracker_id": int(tracker_id),
                "image_path": img_filename,
                "violation_type": f"none_{item}",
                "confidence": float(prob)
            }
            for item, prob in zip(missing_items, missing_probs)
        ]

        data = {
            "violations": violations_list,
            "status": "Warning",
            "session_id": session_id,
        }

        client = get_supabase_client()
        if client:
            client.table("ppe_violations").insert(data).execute()
            logger.info(f"Database sync successful for violation ID: {tracker_id}.")

        # Instant Telegram alert with crop image
        send_telegram_alert(tracker_id, missing_items, crop_img)

    except Exception as e:
        logger.error(f"Database logging execution failed: {e}")