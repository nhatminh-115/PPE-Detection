import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

mock_supabase = MagicMock()
mock_supabase.table.return_value.select.return_value \
    .order.return_value.limit.return_value \
    .execute.return_value.data = []
mock_supabase.table.return_value.insert.return_value.execute.return_value = {}

patch_supabase   = patch("supabase.create_client",              return_value=mock_supabase)
patch_mlflow_dl  = patch("mlflow.artifacts.download_artifacts", return_value="mock_weight.pt")
patch_mlflow_uri = patch("mlflow.set_tracking_uri",             MagicMock())
patch_os_rename  = patch("os.rename",                           MagicMock()) 
patch_torch_load = patch("torch.load",                          return_value={})
patch_yolo       = patch("ultralytics.YOLO",                    MagicMock())
patch_timm       = patch("timm.create_model",                   MagicMock())

patch_supabase.start()
patch_mlflow_dl.start()
patch_mlflow_uri.start()
patch_os_rename.start()
patch_torch_load.start()
patch_yolo.start()
patch_timm.start()

from api import app
client = TestClient(app)

# ==============================================================================
# TEST CASES
# ==============================================================================

def test_read_root():
    """Kiem tra API tra ve trang HTML Index"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_violations_api():
    """Kiem tra endpoint /api/violations tra ve JSON dung format"""
    response = client.get("/api/violations")
    assert response.status_code == 200
    json_data = response.json()
    assert "data" in json_data
    assert isinstance(json_data["data"], list)


def test_flush_images_api():
    """Kiem tra endpoint /api/flush don dep file"""
    response = client.post("/api/flush")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] in ["success", "error"]


def test_upload_video_api_no_file():
    """Kiem tra FastAPI tra 422 khi thieu file upload"""
    response = client.post("/upload_video")
    assert response.status_code == 422


def test_upload_video_api_with_file(tmp_path):
    """Kiem tra upload video thanh cong"""
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake video content")

    with open(fake_video, "rb") as f:
        response = client.post(
            "/upload_video",
            files={"video_file": ("test.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "success"