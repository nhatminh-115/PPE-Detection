import shutil
import logging
from fastapi import Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from src.api.app import app, templates
from src.api.pipeline import generate_frames, stream_state
from src.infrastructure.supabase import get_supabase_client

logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/upload_video")
async def upload_video(video_file: UploadFile = File(...)):
    try:
        file_path = f"temp_uploads/{video_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        stream_state.source = file_path
        stream_state.trigger_restart = True
        return {"status": "success", "message": f"Source injected: {video_file.filename}"}
    except Exception as e:
        logger.error(f"Video injection failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/start_webcam")
def start_webcam(camera_index: int = 0):
    stream_state.source = str(camera_index)
    stream_state.trigger_restart = True
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