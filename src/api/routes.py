import shutil
import logging
from fastapi import Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from src.api.app import app, templates
from src.api.pipeline import generate_frames, stream_state
from src.infrastructure.supabase import get_supabase_client
from src.config import PPE_THRESHOLDS
from src.inference.tracker import reported_violations
from src.inference.classifier import cam_mode_enabled
import src.inference.classifier as _clf
import uuid
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.post("/api/thresholds")
def update_thresholds(
    hardhat_ok: float = 0.70,
    hardhat_warn: float = 0.40,
    vest_ok: float = 0.50,
    vest_warn: float = 0.15,
):
    PPE_THRESHOLDS['hardhat']['ok']   = hardhat_ok
    PPE_THRESHOLDS['hardhat']['warn'] = hardhat_warn
    PPE_THRESHOLDS['vest']['ok']      = vest_ok
    PPE_THRESHOLDS['vest']['warn']    = vest_warn
    return {"status": "success", "thresholds": PPE_THRESHOLDS}

@app.get("/api/thresholds")
def get_thresholds():
    return {"thresholds": PPE_THRESHOLDS}

@app.get("/api/stats")
def get_stats(session_id: str = None):
    client = get_supabase_client()
    if not client:
        return {"total": 0, "none_hardhat": 0, "none_vest": 0, "hourly": {}, "top_offenders": []}

    query = client.table("ppe_violations").select("*")
    if session_id:
        query = query.eq("session_id", session_id)
    res = query.order("created_at", desc=True).limit(200).execute()
    rows = res.data or []

    from collections import defaultdict
    from datetime import datetime, timezone, timedelta
    ICT = timezone(timedelta(hours=7))

    hourly         = defaultdict(int)
    hardhat        = 0
    vest           = 0
    tracker_counts = defaultdict(int)

    for row in rows:
        hour = datetime.fromisoformat(row["created_at"]).astimezone(ICT).hour
        hourly[hour] += 1
        for v in row.get("violations", []):
            if v["violation_type"] == "none_hardhat":
                hardhat += 1
            elif v["violation_type"] == "none_vest":
                vest += 1
            tracker_counts[str(v["tracker_id"])] += 1

    top_offenders = sorted(tracker_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total":         len(rows),
        "none_hardhat":  hardhat,
        "none_vest":     vest,
        "hourly":        dict(hourly),
        "top_offenders": [{"id": k, "count": v} for k, v in top_offenders],
    }

@app.get("/api/sessions")
def get_sessions():
    client = get_supabase_client()
    if not client:
        return {"sessions": []}
    res = (
        client.table("ppe_violations")
        .select("session_id, created_at")
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )
    rows = res.data or []
    seen = {}
    for row in rows:
        sid = row.get("session_id")
        if sid and sid not in seen:
            seen[sid] = row["created_at"]
    return {
        "sessions": [{"id": k, "created_at": v} for k, v in seen.items()]
    }

@app.post("/api/cam_mode")
def toggle_cam_mode(enabled: bool = True):
    _clf.cam_mode_enabled = enabled
    return {"status": "success", "cam_mode": enabled}

@app.post("/upload_video")
async def upload_video(video_file: UploadFile = File(...)):
    try:
        file_path = f"temp_uploads/{video_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        stream_state.source = file_path
        stream_state.trigger_restart = True
        stream_state.session_id      = str(uuid.uuid4())[:8]  # short UUID
        reported_violations.clear()
        return {"status": "success", "message": f"Source injected: {video_file.filename}"}
    except Exception as e:
        logger.error(f"Video injection failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/start_webcam")
def start_webcam(camera_index: int = 0):
    stream_state.source          = str(camera_index)
    stream_state.trigger_restart = True
    stream_state.session_id      = str(uuid.uuid4())[:8]
    reported_violations.clear()
    return {"status": "success", "message": f"Webcam {camera_index} started"}


@app.post("/stop_webcam")
def stop_webcam():
    stream_state.source = None
    stream_state.trigger_restart = True
    return {"status": "success", "message": "Webcam stream stopped"}


@app.get("/api/violations")
def get_violations():
    client = get_supabase_client()
    if not client:
        return {"data": []}
    res = (
        client.table("ppe_violations")
        .select("*")
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    return {"data": res.data}


@app.post("/api/flush")
def flush_images():
    import os
    try:
        count = 0
        for f in os.listdir("violation_crops"):
            file_path = os.path.join("violation_crops", f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        return {"status": "success", "message": f"Flushed {count} objects."}
    except Exception as e:
        logger.error(f"Storage purge failed: {e}")
        return {"status": "error", "message": str(e)}