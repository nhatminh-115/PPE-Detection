import os
import time
import logging
import concurrent.futures
import cv2
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
                logger.error(f"Failed to initialize Supabase: {e}")
        else:
            logger.warning("Supabase credentials missing. Remote logging disabled.")
    return supabase_client

def log_violation_to_supabase(tracker_id, missing_items, missing_probs, crop_img):
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
            "status": "Warning"
        }
        
        client = get_supabase_client()
        if client:
            client.table("ppe_violations").insert(data).execute()
            logger.info(f"Database sync successful for violation ID: {tracker_id}.")
    except Exception as e:
        logger.error(f"Database logging execution failed: {e}")