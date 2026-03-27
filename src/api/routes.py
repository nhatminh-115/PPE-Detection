import shutil
import logging
import uuid
import numpy as np
from fastapi import Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from src.api.app import app, templates
from src.api.pipeline import (
    generate_frames, generate_frames_for_camera,
    stream_state, camera_registry, _latest_frames,
)
from src.infrastructure.supabase import get_supabase_client
from src.config import PPE_THRESHOLDS
from src.inference.tracker import reported_violations
import src.inference.classifier as _clf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


# ---------------------------------------------------------------------------
# Video feeds
# ---------------------------------------------------------------------------

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed/{cam_id}")
def video_feed_camera(cam_id: str):
    return StreamingResponse(generate_frames_for_camera(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Snapshot for zone editor
# ---------------------------------------------------------------------------

@app.get("/snapshot/{cam_id}")
def get_snapshot(cam_id: str):
    import cv2, time as _time
    cam = camera_registry.get(cam_id)
    if not cam:
        return Response(status_code=404)

    # Use cached frame if available
    frame_bytes = _latest_frames.get(cam_id)
    frame_w = cam.frame_w
    frame_h = cam.frame_h

    if not frame_bytes:
        # Grab directly — stream not running yet
        source = cam.source
        if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            _time.sleep(0.5)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return Response(status_code=503)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return Response(status_code=503)

        h, w = frame.shape[:2]
        scale  = 720 / h
        frame  = cv2.resize(frame, (int(w * scale), 720))
        frame_w = int(w * scale)
        frame_h = 720

        ret2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret2:
            return Response(status_code=500)
        frame_bytes = buf.tobytes()

    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "X-Frame-Width":  str(frame_w or 960),
            "X-Frame-Height": str(frame_h or 720),
            "Access-Control-Expose-Headers": "X-Frame-Width, X-Frame-Height",
        },
    )


# ---------------------------------------------------------------------------
# Camera management
# ---------------------------------------------------------------------------

@app.get("/api/cameras")
def list_cameras():
    return {"cameras": camera_registry.list()}


@app.post("/api/cameras/add_webcam")
def add_webcam(camera_index: int = 0, label: str = ""):
    cam = camera_registry.add(source=camera_index, label=label or f"Webcam {camera_index}")
    reported_violations.clear()
    return {"status": "success", "cam_id": cam.cam_id, "label": cam.label}


@app.post("/api/cameras/add_rtsp")
def add_rtsp(url: str, label: str = ""):
    cam = camera_registry.add(source=url, label=label or f"RTSP {url[-20:]}")
    return {"status": "success", "cam_id": cam.cam_id, "label": cam.label}


@app.delete("/api/cameras/{cam_id}")
def remove_camera(cam_id: str):
    ok = camera_registry.remove(cam_id)
    return {"status": "success" if ok else "error"}


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------

@app.post("/api/cameras/{cam_id}/pause")
def pause_camera(cam_id: str):
    cam = camera_registry.get(cam_id)
    if cam:
        cam.paused = True
    return {"status": "ok"}


@app.post("/api/cameras/{cam_id}/resume")
def resume_camera(cam_id: str):
    cam = camera_registry.get(cam_id)
    if cam:
        cam.paused = False
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Zone management
# ---------------------------------------------------------------------------

@app.get("/api/cameras/{cam_id}/zone")
def get_zone(cam_id: str):
    cam = camera_registry.get(cam_id)
    if not cam:
        return {"status": "error"}
    if cam.zone_polygon is None:
        return {"status": "ok", "zone": None, "frame_w": cam.frame_w or 960, "frame_h": cam.frame_h or 720}

    zone = cam.zone_polygon.astype(float)
    disp_w = float(cam.frame_w or 0)
    disp_h = float(cam.frame_h or 0)
    proc_w = float(cam.proc_w or 0)
    proc_h = float(cam.proc_h or 0)

    # Backend stores zone in processing-frame coordinates; convert to display-frame for editor.
    if disp_w > 0 and disp_h > 0 and proc_w > 0 and proc_h > 0:
        sx = disp_w / proc_w
        sy = disp_h / proc_h
        zone = zone.copy()
        zone[:, 0] *= sx
        zone[:, 1] *= sy

    return {
        "status": "ok",
        "zone": zone.round().astype(int).tolist(),
        "frame_w": cam.frame_w or 960,
        "frame_h": cam.frame_h or 720,
    }


@app.post("/api/cameras/{cam_id}/zone")
def set_zone(cam_id: str, points: list[list[float]]):
    cam = camera_registry.get(cam_id)
    if not cam:
        return {"status": "error"}
    if len(points) < 3:
        return {"status": "error", "message": "Need at least 3 points"}

    zone = np.array(points, dtype=np.float32)
    disp_w = float(cam.frame_w or 0)
    disp_h = float(cam.frame_h or 0)
    proc_w = float(cam.proc_w or 0)
    proc_h = float(cam.proc_h or 0)

    # Editor sends display-frame coordinates; convert to processing-frame for filtering.
    if disp_w > 0 and disp_h > 0 and proc_w > 0 and proc_h > 0:
        sx = proc_w / disp_w
        sy = proc_h / disp_h
        zone[:, 0] *= sx
        zone[:, 1] *= sy

    cam.zone_polygon = zone.round().astype(np.int32).copy()
    return {"status": "ok"}


@app.delete("/api/cameras/{cam_id}/zone")
def delete_zone(cam_id: str):
    cam = camera_registry.get(cam_id)
    if cam:
        cam.zone_polygon = None
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Legacy single-camera controls
# ---------------------------------------------------------------------------

@app.post("/upload_video")
async def upload_video(video_file: UploadFile = File(...)):
    try:
        file_path = f"temp_uploads/{video_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # Multi-camera flow: inject uploaded file as a real camera source.
        cam = camera_registry.add(source=file_path, label=video_file.filename or "Uploaded Video")

        # Keep legacy state in sync for backward compatibility.
        stream_state.source          = file_path
        stream_state.trigger_restart = True
        stream_state.session_id      = str(uuid.uuid4())[:8]
        reported_violations.clear()
        return {
            "status": "success",
            "message": f"Source injected: {video_file.filename}",
            "cam_id": cam.cam_id,
            "label": cam.label,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/start_webcam")
def start_webcam(camera_index: int = 0):
    stream_state.source          = str(camera_index)
    stream_state.trigger_restart = True
    stream_state.session_id      = str(uuid.uuid4())[:8]
    reported_violations.clear()
    return {"status": "success"}


@app.post("/stop_webcam")
def stop_webcam():
    stream_state.source          = None
    stream_state.trigger_restart = True
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@app.post("/api/thresholds")
def update_thresholds(hardhat_ok: float = 0.70, hardhat_warn: float = 0.40, vest_ok: float = 0.50, vest_warn: float = 0.15):
    PPE_THRESHOLDS['hardhat']['ok']   = hardhat_ok
    PPE_THRESHOLDS['hardhat']['warn'] = hardhat_warn
    PPE_THRESHOLDS['vest']['ok']      = vest_ok
    PPE_THRESHOLDS['vest']['warn']    = vest_warn
    return {"status": "success", "thresholds": PPE_THRESHOLDS}


@app.get("/api/thresholds")
def get_thresholds():
    return {"thresholds": PPE_THRESHOLDS}


@app.get("/api/router_config")
def get_router_config():
    return {
        "min_side_px": float(_clf.router_min_side_px),
    }


@app.post("/api/router_config")
def update_router_config(min_side_px: float = 110.0):
    # Keep threshold in a practical range to avoid accidental extremes from UI.
    clamped = max(32.0, min(512.0, float(min_side_px)))
    _clf.router_min_side_px = clamped
    return {
        "status": "success",
        "router": {
            "min_side_px": clamped,
        },
    }


# ---------------------------------------------------------------------------
# CAM mode
# ---------------------------------------------------------------------------

@app.post("/api/cam_mode")
def toggle_cam_mode(enabled: bool = True):
    _clf.cam_mode_enabled = enabled
    return {"status": "success", "cam_mode": enabled}


@app.get("/api/siglip_pose")
def get_siglip_pose_mode():
    enabled = bool(_clf.siglip_use_pose)
    return {"siglip_use_pose": enabled, "clip_use_pose": enabled}


@app.post("/api/siglip_pose")
def set_siglip_pose_mode(enabled: bool = True):
    _clf.siglip_use_pose = bool(enabled)
    return {
        "status": "success",
        "siglip_use_pose": bool(_clf.siglip_use_pose),
        "clip_use_pose": bool(_clf.siglip_use_pose),
    }


# Backward-compatible aliases.
@app.get("/api/clip_pose")
def get_clip_pose_mode():
    return get_siglip_pose_mode()


@app.post("/api/clip_pose")
def set_clip_pose_mode(enabled: bool = True):
    return set_siglip_pose_mode(enabled)


# ---------------------------------------------------------------------------
# Stats & violations
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def get_stats(session_id: str = None):
    client = get_supabase_client()
    if not client:
        return {"total": 0, "none_hardhat": 0, "none_vest": 0, "hourly": {}, "top_offenders": []}
    query = client.table("ppe_violations").select("*")
    if session_id:
        query = query.eq("session_id", session_id)
    rows = query.order("created_at", desc=True).limit(200).execute().data or []

    from collections import defaultdict
    from datetime import datetime, timezone, timedelta
    ICT = timezone(timedelta(hours=7))
    hourly, hardhat, vest, tc = defaultdict(int), 0, 0, defaultdict(int)
    for row in rows:
        hourly[datetime.fromisoformat(row["created_at"]).astimezone(ICT).hour] += 1
        for v in row.get("violations", []):
            if v["violation_type"] == "none_hardhat": hardhat += 1
            elif v["violation_type"] == "none_vest":  vest += 1
            tc[str(v["tracker_id"])] += 1
    top = sorted(tc.items(), key=lambda x: x[1], reverse=True)[:5]
    return {"total": len(rows), "none_hardhat": hardhat, "none_vest": vest, "hourly": dict(hourly), "top_offenders": [{"id": k, "count": v} for k, v in top]}


@app.get("/api/sessions")
def get_sessions():
    client = get_supabase_client()
    if not client:
        return {"sessions": []}
    rows = client.table("ppe_violations").select("session_id, created_at").order("created_at", desc=True).limit(200).execute().data or []
    seen = {}
    for row in rows:
        sid = row.get("session_id")
        if sid and sid not in seen:
            seen[sid] = row["created_at"]
    return {"sessions": [{"id": k, "created_at": v} for k, v in seen.items()]}


@app.get("/api/violations")
def get_violations():
    client = get_supabase_client()
    if not client:
        return {"data": []}
    res = client.table("ppe_violations").select("*").order("created_at", desc=True).limit(20).execute()
    return {"data": res.data}


@app.post("/api/flush")
def flush_images():
    import os
    try:
        count = sum(1 for f in os.listdir("violation_crops") if os.path.isfile(os.path.join("violation_crops", f)) and not os.remove(os.path.join("violation_crops", f)))
        return {"status": "success", "message": f"Flushed {count} objects."}
    except Exception as e:
        return {"status": "error", "message": str(e)}