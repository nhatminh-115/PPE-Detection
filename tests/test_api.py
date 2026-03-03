import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ==============================================================================
# MOCK DEPENDENCIES
# Thứ tự quan trọng:
# 1. patch supabase.create_client TRƯỚC — ngăn init_supabase() crash ở global scope
# 2. patch torch.load TRƯỚC — ngăn load model thật
# 3. Sau đó mới import api
# ==============================================================================

mock_supabase = MagicMock()
mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = []
mock_supabase.table.return_value.insert.return_value.execute.return_value = {}

patch_supabase_create = patch("supabase.create_client", return_value=mock_supabase)
patch_yolo             = patch("ultralytics.YOLO", MagicMock())
patch_timm             = patch("timm.create_model", MagicMock())
patch_torch_load       = patch("torch.load", return_value={})
patch_mlflow           = patch("mlflow.set_tracking_uri", MagicMock())
patch_pull             = patch("api.pull_artifact_from_mlflow_run", return_value="mock_weight.pt")

patch_supabase_create.start()
patch_yolo.start()
patch_timm.start()
patch_torch_load.start()
patch_mlflow.start()
patch_pull.start()

# Import SAU KHI tất cả patch đã active
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