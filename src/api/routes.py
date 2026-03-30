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
from postgrest.exceptions import APIError as SupabaseAPIError
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

    # Stream has a cached frame but dimensions not yet written (tight race on
    # first frame).  Decode to get exact size.
    if frame_bytes and (not frame_w or not frame_h):
        import numpy as _np
        arr = _np.frombuffer(frame_bytes, dtype=_np.uint8)
        tmp = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if tmp is not None:
            fh, fw = tmp.shape[:2]
            frame_w = fw
            frame_h = fh
            cam.frame_w = frame_w
            cam.frame_h = frame_h

    if not frame_bytes:
        # Grab directly — stream not running yet
        source = cam.source
        if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
            cap = cv2.VideoCapture(int(source), cv2.CAP_MSMF)
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
        scale   = 720 / h
        frame   = cv2.resize(frame, (int(w * scale), 720))
        frame_w = int(w * scale)
        frame_h = 720

        # Persist dimensions so set_zone coordinate conversion works correctly
        # even when the user opens the zone editor before the stream has
        # processed its first frame.
        cam.proc_w  = w        # raw camera width  (= inference frame width)
        cam.proc_h  = h        # raw camera height (= inference frame height)
        cam.frame_w = frame_w  # display width (720p-scaled)
        cam.frame_h = frame_h

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
def update_thresholds(
    hardhat_ok: float = float(PPE_THRESHOLDS['hardhat']['ok']),
    hardhat_warn: float = float(PPE_THRESHOLDS['hardhat']['warn']),
    vest_ok: float = float(PPE_THRESHOLDS['vest']['ok']),
    vest_warn: float = float(PPE_THRESHOLDS['vest']['warn']),
):
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
    try:
        rows = query.order("created_at", desc=True).limit(200).execute().data or []
    except (SupabaseAPIError, Exception) as exc:
        logger.warning(f"Supabase unavailable (stats): {exc}")
        return {"total": 0, "none_hardhat": 0, "none_vest": 0, "hourly": {}, "top_offenders": []}

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
    try:
        rows = client.table("ppe_violations").select("session_id, created_at").order("created_at", desc=True).limit(200).execute().data or []
    except (SupabaseAPIError, Exception) as exc:
        logger.warning(f"Supabase unavailable (sessions): {exc}")
        return {"sessions": []}
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
    try:
        res = client.table("ppe_violations").select("*").order("created_at", desc=True).limit(20).execute()
        return {"data": res.data}
    except (SupabaseAPIError, Exception) as exc:
        logger.warning(f"Supabase unavailable (violations): {exc}")
        return {"data": []}


@app.get("/api/report/pdf")
def download_report_pdf(session_id: str = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import io
    from datetime import datetime, timezone, timedelta
    from collections import defaultdict

    client = get_supabase_client()
    if not client:
        return Response(status_code=503)

    ICT = timezone(timedelta(hours=7))
    now = datetime.now(ICT)

    query = client.table("ppe_violations").select("*")
    if session_id:
        query = query.eq("session_id", session_id)
    try:
        rows = query.order("created_at", desc=True).limit(500).execute().data or []
    except (SupabaseAPIError, Exception) as exc:
        logger.warning(f"Supabase unavailable (report): {exc}")
        from fastapi.responses import Response as _Resp
        return _Resp(status_code=503, content="Database temporarily unavailable")

    hourly: dict = defaultdict(int)
    hardhat_count = 0
    vest_count = 0
    tc: dict = defaultdict(int)
    for row in rows:
        dt = datetime.fromisoformat(row["created_at"]).astimezone(ICT)
        hourly[dt.hour] += 1
        for v in row.get("violations", []):
            if v["violation_type"] == "none_hardhat":
                hardhat_count += 1
            elif v["violation_type"] == "none_vest":
                vest_count += 1
            tc[str(v["tracker_id"])] += 1

    top_offenders = sorted(tc.items(), key=lambda x: x[1], reverse=True)[:8]
    peak_hour = max(hourly, key=hourly.get) if hourly else None
    total = len(rows)

    BG       = '#ffffff'
    HDR_BG   = '#0d1117'
    ACCENT   = '#388bfd'
    DANGER   = '#f85149'
    WARNING  = '#f59e0b'
    SUCCESS  = '#3fb950'
    TXT_D    = '#1a1a2e'
    TXT_M    = '#57606a'
    BORDER   = '#d0d7de'

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(BG)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = fig.add_axes([0, 0.930, 1, 0.070])
        hdr.set_facecolor(HDR_BG)
        hdr.axis('off')
        hdr.text(0.030, 0.66, 'PPE VISION', fontsize=15, fontweight='bold',
                 color='white', va='center', transform=hdr.transAxes)
        hdr.text(0.030, 0.24, 'SAFETY COMPLIANCE REPORT',
                 fontsize=8, color='#8b949e', va='center',
                 transform=hdr.transAxes, fontstyle='normal')
        session_label = f"Session: {session_id}" if session_id else "All Sessions"
        hdr.text(0.972, 0.66,
                 f"Generated: {now.strftime('%d %b %Y, %H:%M')} ICT",
                 fontsize=8, color='#8b949e', va='center', ha='right',
                 transform=hdr.transAxes)
        hdr.text(0.972, 0.24,
                 f"{session_label}  ·  {total} records",
                 fontsize=8, color='#8b949e', va='center', ha='right',
                 transform=hdr.transAxes)

        acc = fig.add_axes([0, 0.927, 1, 0.003])
        acc.set_facecolor(ACCENT)
        acc.axis('off')

        # ── KPI cards ───────────────────────────────────────────────────────
        top_id = f"ID {top_offenders[0][0]}" if top_offenders else '—'
        peak_lbl = f"{str(peak_hour).zfill(2)}:00" if peak_hour is not None else '—'
        kpis = [
            ('TOTAL EVENTS',     str(total),         TXT_D,   '#f6f8fa'),
            ('MISSING HARDHAT',  str(hardhat_count), DANGER,  '#fff0f0'),
            ('MISSING VEST',     str(vest_count),    WARNING, '#fffbea'),
            ('PEAK HOUR (ICT)',  peak_lbl,           SUCCESS, '#f0fff4'),
            ('TOP OFFENDER',     top_id,             ACCENT,  '#f0f6ff'),
        ]
        cw, cgap = 0.184, 0.005
        for i, (lbl, val, clr, bg) in enumerate(kpis):
            x = 0.020 + i * (cw + cgap)
            ca = fig.add_axes([x, 0.845, cw, 0.077])
            ca.set_facecolor(bg)
            ca.axis('off')
            for sp in ca.spines.values():
                sp.set_visible(False)
            rect = mpatches.FancyBboxPatch(
                (0.01, 0.01), 0.98, 0.98,
                boxstyle="round,pad=0.01", linewidth=0.6,
                edgecolor=BORDER, facecolor='none',
                transform=ca.transAxes, clip_on=False,
            )
            ca.add_patch(rect)
            ca.text(0.50, 0.64, val, fontsize=17, fontweight='bold',
                    color=clr, ha='center', va='center', transform=ca.transAxes)
            ca.text(0.50, 0.20, lbl, fontsize=6.5, color=TXT_M,
                    ha='center', va='center', transform=ca.transAxes)

        # ── Hourly bar chart ─────────────────────────────────────────────────
        ha = fig.add_axes([0.055, 0.580, 0.555, 0.235])
        ha.set_facecolor('#f6f8fa')
        vals = [hourly.get(h, 0) for h in range(24)]
        bar_clrs = [
            WARNING if h == peak_hour else (DANGER if v > 0 else '#d0d7de')
            for h, v in enumerate(vals)
        ]
        ha.bar(range(24), vals, color=bar_clrs, width=0.72, edgecolor='none', zorder=3)
        ha.set_xlim(-0.5, 23.5)
        ha.set_ylim(0, max(max(vals) * 1.25, 1))
        ha.set_xticks([0, 6, 12, 18, 23])
        ha.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'],
                           fontsize=7, color=TXT_M)
        ha.tick_params(axis='y', labelsize=7, colors=TXT_M)
        ha.grid(axis='y', color=BORDER, linewidth=0.5, zorder=0)
        ha.set_axisbelow(True)
        for sp in ['top', 'right']:
            ha.spines[sp].set_visible(False)
        ha.spines['left'].set_color(BORDER)
        ha.spines['bottom'].set_color(BORDER)
        ha.set_title('Violations by Hour (ICT)', fontsize=9, fontweight='600',
                     color=TXT_D, pad=6, loc='left')
        legend_patches = [
            mpatches.Patch(color=DANGER,  label='Violations'),
            mpatches.Patch(color=WARNING, label='Peak hour'),
        ]
        ha.legend(handles=legend_patches, fontsize=7, loc='upper right',
                  framealpha=0.9, edgecolor=BORDER)

        # ── Donut chart ──────────────────────────────────────────────────────
        da = fig.add_axes([0.655, 0.600, 0.300, 0.210])
        da.set_facecolor(BG)
        da.set_aspect('equal')
        total_types = hardhat_count + vest_count
        if total_types > 0:
            sizes = [hardhat_count, vest_count]
            wedges, _ = da.pie(sizes, colors=[DANGER, WARNING], startangle=90,
                               wedgeprops=dict(width=0.46))
            da.text(0, 0, str(total_types), ha='center', va='center',
                    fontsize=13, fontweight='bold', color=TXT_D)
        else:
            da.pie([1], colors=['#d0d7de'], wedgeprops=dict(width=0.46))
            da.text(0, 0, '0', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=TXT_M)
        da.set_title('Violation Type', fontsize=9, fontweight='600',
                     color=TXT_D, pad=6, loc='left')

        hh_pct = f"{round(hardhat_count/total_types*100)}%" if total_types else "0%"
        vt_pct = f"{round(vest_count/total_types*100)}%" if total_types else "0%"
        la = fig.add_axes([0.655, 0.565, 0.300, 0.038])
        la.axis('off')
        la.text(0.00, 0.75, '●', color=DANGER, fontsize=9,
                transform=la.transAxes, va='center')
        la.text(0.10, 0.75, f'No Hardhat — {hardhat_count} ({hh_pct})',
                fontsize=7, color=TXT_M, transform=la.transAxes, va='center')
        la.text(0.00, 0.20, '●', color=WARNING, fontsize=9,
                transform=la.transAxes, va='center')
        la.text(0.10, 0.20, f'No Vest — {vest_count} ({vt_pct})',
                fontsize=7, color=TXT_M, transform=la.transAxes, va='center')

        # ── Top offenders ────────────────────────────────────────────────────
        oa = fig.add_axes([0.055, 0.415, 0.900, 0.140])
        oa.set_facecolor('#f6f8fa')
        if top_offenders:
            ids_lbl = [f'ID {t[0]}' for t in top_offenders]
            counts   = [t[1] for t in top_offenders]
            y_pos    = list(range(len(top_offenders)))
            bars_o   = oa.barh(y_pos, counts, color=DANGER, alpha=0.80,
                               height=0.55, edgecolor='none')
            for bar, cnt in zip(bars_o, counts):
                oa.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                        str(cnt), va='center', fontsize=7, color=TXT_D, fontweight='600')
            oa.set_yticks(y_pos)
            oa.set_yticklabels(ids_lbl, fontsize=7.5, color=TXT_D)
            oa.tick_params(axis='x', labelsize=7, colors=TXT_M)
            oa.set_xlim(0, max(counts) * 1.18)
            oa.grid(axis='x', color=BORDER, linewidth=0.5, zorder=0)
            oa.set_axisbelow(True)
            oa.invert_yaxis()
        else:
            oa.text(0.5, 0.5, 'No violations recorded', fontsize=9,
                    color=TXT_M, ha='center', va='center', transform=oa.transAxes)
        for sp in ['top', 'right']:
            oa.spines[sp].set_visible(False)
        oa.spines['left'].set_color(BORDER)
        oa.spines['bottom'].set_color(BORDER)
        oa.set_title('Top Repeat Offenders', fontsize=9, fontweight='600',
                     color=TXT_D, pad=6, loc='left')

        # ── Violations table ─────────────────────────────────────────────────
        MAX_ROWS = 15
        table_rows = rows[:MAX_ROWS]
        ROW_H = 0.052   # in axes coordinates
        HDR_H = 0.060
        TITLE_H = 0.042
        table_ax_h = TITLE_H + HDR_H + len(table_rows) * ROW_H + 0.04
        ta = fig.add_axes([0.030, 0.030, 0.940, min(table_ax_h, 0.370)])
        ta.axis('off')
        ta.set_facecolor(BG)

        top_y = 1.0 - TITLE_H / ta.get_position().height * ta.get_position().height
        ta.text(0.0, 0.985,
                f'Recent Violation Log  (latest {min(len(rows), MAX_ROWS)} of {total})',
                fontsize=9, fontweight='600', color=TXT_D,
                transform=ta.transAxes, va='top')

        cols      = ['#', 'Timestamp (ICT)', 'Tracker', 'Violations', 'Model', 'Session']
        col_x     = [0.000, 0.040, 0.195, 0.290, 0.560, 0.680]
        col_end_x = [0.038, 0.192, 0.286, 0.556, 0.676, 0.800]

        hdr_y = 0.920
        for lbl, x0, x1 in zip(cols, col_x, col_end_x):
            ta.add_patch(mpatches.Rectangle(
                (x0, hdr_y - HDR_H * 0.85), x1 - x0 - 0.003, HDR_H * 0.85,
                facecolor=HDR_BG, transform=ta.transAxes, clip_on=False,
            ))
            ta.text(x0 + 0.006, hdr_y - HDR_H * 0.38, lbl,
                    fontsize=6.5, color='white', fontweight='600',
                    transform=ta.transAxes, va='center')

        for idx, row in enumerate(table_rows):
            row_y = hdr_y - HDR_H * 0.85 - (idx + 1) * ROW_H * 0.88
            if row_y < 0:
                break
            row_bg = '#f6f8fa' if idx % 2 == 0 else BG
            for x0, x1 in zip(col_x, col_end_x):
                ta.add_patch(mpatches.Rectangle(
                    (x0, row_y), x1 - x0 - 0.003, ROW_H * 0.82,
                    facecolor=row_bg, edgecolor=BORDER, linewidth=0.3,
                    transform=ta.transAxes, clip_on=False,
                ))
            dt_row  = datetime.fromisoformat(row["created_at"]).astimezone(ICT)
            viols   = row.get("violations", [])
            viol_str = ', '.join(dict.fromkeys(
                v["violation_type"].replace("none_", "").upper() for v in viols
            ))
            t_id  = str(viols[0]["tracker_id"]) if viols else '—'
            model = (row.get("reported_by_model") or
                     (viols[0].get("reported_by_model", "?") if viols else "?"))
            model = str(model).upper()[:8]
            sess  = str(row.get("session_id", ""))[:8]
            cells = [str(idx+1), dt_row.strftime('%m/%d %H:%M:%S'),
                     t_id, viol_str, model, sess]
            for val, x0 in zip(cells, col_x):
                txt_clr = DANGER if any(k in val for k in ('HARDHAT', 'VEST')) else TXT_D
                ta.text(x0 + 0.006, row_y + ROW_H * 0.37,
                        str(val)[:22], fontsize=6, color=txt_clr,
                        transform=ta.transAxes, va='center')

        # ── Footer ───────────────────────────────────────────────────────────
        fa = fig.add_axes([0, 0.000, 1, 0.026])
        fa.set_facecolor('#f6f8fa')
        fa.axis('off')
        fa.axhline(y=0.98, color=BORDER, linewidth=0.5)
        fa.text(0.030, 0.50, 'PPE Vision — Operations Center',
                fontsize=7, color=TXT_M, va='center', transform=fa.transAxes)
        fa.text(0.970, 0.50,
                f'Generated {now.strftime("%Y-%m-%d %H:%M:%S")} ICT  ·  Page 1 of 1',
                fontsize=7, color=TXT_M, va='center', ha='right',
                transform=fa.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    buf.seek(0)
    filename = f"ppe_report_{now.strftime('%Y%m%d_%H%M')}.pdf"
    return Response(
        content=buf.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/flush")
def flush_images():
    import os
    try:
        count = sum(1 for f in os.listdir("violation_crops") if os.path.isfile(os.path.join("violation_crops", f)) and not os.remove(os.path.join("violation_crops", f)))
        return {"status": "success", "message": f"Flushed {count} objects."}
    except Exception as e:
        return {"status": "error", "message": str(e)}