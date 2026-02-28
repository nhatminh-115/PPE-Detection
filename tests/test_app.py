import sys
import os
import pytest
import io
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. KỸ THUẬT TIỀN TRẠM ĐÁNH CHẶN (PRE-IMPORT PATCHING)
# Bịt kín mọi đường ra Internet và VRAM trước khi file app.py được cấp phép thực thi
patch('mlflow.tracking.MlflowClient').start()
patch('mlflow.artifacts.download_artifacts').start()
patch('ultralytics.YOLO').start()
patch('supabase.create_client').start()

# 2. Bây giờ mới import. Toàn bộ hàm initialize_model bên trong sẽ chỉ chọc vào các biến Dummy (0.001 giây)
import app
from app import app as flask_app

# 3. Ép trạng thái của AI là "đang sống" để các hàm test đi sâu được vào luồng kiểm tra hình ảnh
app.model = MagicMock()

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_missing_image_payload(client):
    """Kỳ vọng: Lỗi 400 do không có ảnh đính kèm."""
    response = client.post('/detect')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "payload is missing" in data["error"]

def test_invalid_image_format(client):
    """Kỳ vọng: Lỗi 400 do file upload lên là file rác, không giải mã được."""
    dummy_file = (io.BytesIO(b"this_is_not_a_valid_image_byte_stream"), 'test.jpg')
    
    response = client.post(
        '/detect',
        data={'file': dummy_file},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Invalid image format" in data["error"]

def test_engine_uninitialized_graceful_handling(client):
    """Kỳ vọng: Báo lỗi 503 nếu AI Engine bị crash (None)."""
    # Tạm thời phá hủy đồ thị AI để kiểm tra cơ chế phòng ngự 503
    original_model = app.model
    app.model = None 
    
    dummy_image = (io.BytesIO(b"fake_image"), 'test.jpg')
    response = client.post(
        '/detect',
        data={'file': dummy_image},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 503
    data = json.loads(response.data)
    assert "error" in data
    assert "Inference engine is not initialized" in data["error"]
    
    # Phục hồi trạng thái an toàn để không ảnh hưởng các test phía sau
    app.model = original_model