import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.modules["torch"]                     = MagicMock()
sys.modules["torch.nn"]                  = MagicMock()
sys.modules["ultralytics"]               = MagicMock()
sys.modules["timm"]                      = MagicMock()
sys.modules["cv2"]                       = MagicMock()
sys.modules["ensemble_boxes"]            = MagicMock()

mock_supabase = MagicMock()
mock_supabase.table.return_value.select.return_value \
    .order.return_value.limit.return_value \
    .execute.return_value.data = []
mock_supabase.table.return_value.insert.return_value.execute.return_value = {}

patch_supabase   = patch("supabase.create_client",              return_value=mock_supabase)
patch_mlflow_dl  = patch("mlflow.artifacts.download_artifacts", return_value="mock_weight.pt")
patch_mlflow_uri = patch("mlflow.set_tracking_uri",             MagicMock())
patch_os_rename  = patch("os.rename",                           MagicMock())

patch_supabase.start()
patch_mlflow_dl.start()
patch_mlflow_uri.start()
patch_os_rename.start()

import src.api.routes
from src.api.app import app
client = TestClient(app)

# test cases giu nguyen...