import os
import sys
import pytest
from unittest.mock import patch, MagicMock

_MOCK_MODULES = [
    "torch", "torch.nn", "torch.nn.functional",
    "torchvision", "torchvision.transforms", "torchvision.models",
    "torchmetrics",
    "ultralytics",
    "timm", "timm.models",
    "cv2",
    "ensemble_boxes",
    "supervision",
    "lap",
    "sahi",
    "mlflow", "mlflow.artifacts",
    "supabase",
]
for mod in _MOCK_MODULES:
    sys.modules[mod] = MagicMock()

mock_supabase_client = MagicMock()
mock_supabase_client.table.return_value.select.return_value \
    .order.return_value.limit.return_value \
    .execute.return_value.data = []
mock_supabase_client.table.return_value.insert.return_value.execute.return_value = {}
sys.modules["supabase"].create_client = MagicMock(return_value=mock_supabase_client)

# Set env vars
os.environ.setdefault("SUPABASE_URL", "https://mock.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "mock_key")
os.environ.setdefault("MLFLOW_TRACKING_URI", "https://mock.dagshub.com")

# Patch os.rename truoc khi import app
with patch("os.rename", MagicMock()):
    import src.api.routes
    from src.api.app import app

from fastapi.testclient import TestClient
client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_violations_api():
    response = client.get("/api/violations")
    assert response.status_code == 200
    json_data = response.json()
    assert "data" in json_data
    assert isinstance(json_data["data"], list)


def test_flush_images_api():
    response = client.post("/api/flush")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] in ["success", "error"]


def test_upload_video_api_no_file():
    response = client.post("/upload_video")
    assert response.status_code == 422


def test_upload_video_api_with_file(tmp_path):
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake video content")
    with open(fake_video, "rb") as f:
        response = client.post(
            "/upload_video",
            files={"video_file": ("test.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "success"