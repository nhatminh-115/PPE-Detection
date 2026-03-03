import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# 1. MOCK DEPENDENCIES: Ngăn chặn tải Model nặng và Database thật
# Phải patch trước khi import api.py để chặn việc thực thi ở Global Scope
patch_yolo = patch("ultralytics.YOLO", MagicMock())
patch_timm = patch("timm.create_model", MagicMock())
patch_pull = patch("api.pull_artifact_from_mlflow_run", return_value="mock_weight.pt")
patch_db   = patch("api.supabase_client", MagicMock())

patch_yolo.start()
patch_timm.start()
patch_pull.start()
patch_db.start()

# 2. IMPORT MODULE SAU KHI ĐÃ BỌC MOCK
from api import app
client = TestClient(app)

# ==============================================================================
# TEST CASES SUITE
# ==============================================================================

def test_read_root():
    """Kiem tra API co phan hoi trang HTML Index khong"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_get_violations_api():
    """Kiem tra Endpoint cung cap du lieu Supabase"""
    response = client.get("/api/violations")
    assert response.status_code == 200
    json_data = response.json()
    assert "data" in json_data
    # Vi data bi mock, nen no se tra ve list rong
    assert isinstance(json_data["data"], list)

def test_flush_images_api():
    """Kiem tra chuc nang Don dep rac (Garbage Collection)"""
    response = client.post("/api/flush")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] in ["success", "error"]

def test_upload_video_api_no_file():
    """Kiem tra phan ung cua he thong khi gui form ma thieu file"""
    response = client.post("/upload_video")
    # Vi FastAPI bat loi thieu Validation, no phai tra ve 422 Unprocessable Entity
    assert response.status_code == 422