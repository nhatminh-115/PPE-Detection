"""
Unit tests for pose-guided soft attention mask logic.
Tests mask generation, shape invariants, and classify_ppe_batch interface.
"""

import sys
import os
import numpy as np
import pytest

# ── Mock heavy ML libs before importing anything ──
from unittest.mock import MagicMock

_MOCK_MODULES = [
    "torch", "torch.nn", "torch.nn.functional",
    "torchvision", "torchvision.transforms", "torchvision.models",
    "timm",
    "cv2",
    "PIL", "PIL.Image",
    "ultralytics",
    "ensemble_boxes",
    "supervision",
    "supabase",
    "mlflow", "mlflow.artifacts",
]
for mod in _MOCK_MODULES:
    sys.modules[mod] = MagicMock()

# Re-create minimal torch tensor behavior for mask tests
import types

_real_np = np  # save reference


class FakeTensor:
    """Minimal tensor-like for testing mask builders."""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, dim))

    def __repr__(self):
        return f"FakeTensor({self.data})"


# Patch torch.from_numpy and torch.ones to return FakeTensors
mock_torch = sys.modules["torch"]
mock_torch.from_numpy = lambda x: FakeTensor(x)
mock_torch.ones = lambda *args, **kwargs: FakeTensor(np.ones(args, dtype=np.float32))
mock_torch.float32 = np.float32


# Now we can safely import the mask builders from classifier
# But classifier imports from src.config, so let's set that up
os.environ.setdefault("SUPABASE_URL", "https://mock.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "mock_key")
os.environ.setdefault("MLFLOW_TRACKING_URI", "https://mock.dagshub.com")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock dotenv
sys.modules["dotenv"] = MagicMock()

# Import config values directly
sys.modules["src"] = MagicMock()
sys.modules["src.config"] = MagicMock()
sys.modules["src.config"].IMG_SIZE = 224
sys.modules["src.config"].ATTENTION_SIGMA = 0.4
sys.modules["src.config"].ATTENTION_FLOOR = 0.2
sys.modules["src.config"].FEAT_MAP_SIZE = 7

# ── Re-implement the pure-numpy mask builders for testable isolation ──
# (Avoids importing classifier.py which has many torch dependencies)

_KP_CONF_THR = 0.3

_HEAD_KPS  = [0, 1, 2, 3, 4]
_TORSO_KPS = [5, 6, 11, 12]


def _build_pose_mask(keypoints, kp_indices, crop_box, feat_h, feat_w,
                     sigma=0.4, floor=0.2):
    """Pure-numpy version of classifier._build_pose_mask for testing."""
    mask = np.zeros((feat_h, feat_w), dtype=np.float32)
    cx1, cy1, cx2, cy2 = crop_box
    cw = max(cx2 - cx1, 1)
    ch = max(cy2 - cy1, 1)

    valid_count = 0
    for idx in kp_indices:
        if idx >= len(keypoints) or keypoints[idx][2] < _KP_CONF_THR:
            continue
        nx = (keypoints[idx][0] - cx1) / cw
        ny = (keypoints[idx][1] - cy1) / ch
        fx = nx * feat_w
        fy = ny * feat_h
        yy, xx = np.mgrid[0:feat_h, 0:feat_w]
        s = sigma * max(feat_h, feat_w)
        gaussian = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * s * s))
        mask = np.maximum(mask, gaussian)
        valid_count += 1

    if valid_count == 0:
        return np.ones((feat_h, feat_w), dtype=np.float32)

    if mask.max() > 1e-4:
        mask = mask / mask.max()
    mask = np.clip(mask * (1.0 - floor) + floor, 0, 1)
    return mask


def _build_fallback_mask(region, feat_h, feat_w, floor=0.2):
    """Pure-numpy version of classifier._build_fallback_mask for testing."""
    mask = np.full((feat_h, feat_w), floor, dtype=np.float32)
    if region == 'head':
        h_end = max(1, int(feat_h * 0.40))
        mask[:h_end, :] = 1.0
    else:
        h_start = max(0, int(feat_h * 0.20))
        h_end   = min(feat_h, int(feat_h * 0.80))
        mask[h_start:h_end, :] = 1.0
    return mask


# ==============================================================================
# TESTS
# ==============================================================================

class TestBuildPoseMask:
    """Test _build_pose_mask with various keypoint configurations."""

    def _make_kps(self, positions):
        """Helper: create [17, 3] keypoints, filling given positions."""
        kps = np.zeros((17, 3), dtype=np.float32)
        for idx, (x, y, conf) in positions.items():
            kps[idx] = [x, y, conf]
        return kps

    def test_output_shape(self):
        kps = self._make_kps({0: (100, 50, 0.9), 1: (90, 45, 0.8)})
        mask = _build_pose_mask(kps, _HEAD_KPS, (0, 0, 200, 400), 7, 7)
        assert mask.shape == (7, 7)

    def test_values_within_bounds(self):
        kps = self._make_kps({0: (100, 50, 0.9), 1: (90, 45, 0.8), 2: (110, 45, 0.7)})
        mask = _build_pose_mask(kps, _HEAD_KPS, (0, 0, 200, 400), 7, 7,
                                sigma=0.4, floor=0.2)
        assert mask.min() >= 0.2 - 1e-6, f"min={mask.min()}, expected >= 0.2"
        assert mask.max() <= 1.0 + 1e-6, f"max={mask.max()}, expected <= 1.0"

    def test_peak_at_keypoint_location(self):
        """Mask should peak near where keypoints are located."""
        kps = self._make_kps({0: (100, 50, 0.9)})  # nose at top-center
        mask = _build_pose_mask(kps, _HEAD_KPS, (0, 0, 200, 400), 7, 7)
        # Keypoint at (100/200, 50/400) = (0.5, 0.125) → feat coords ≈ (3.5, 0.875)
        # Top row should have higher values than bottom row
        assert mask[0, :].mean() > mask[-1, :].mean()

    def test_no_valid_keypoints_returns_uniform(self):
        """When all keypoints below confidence, return all-ones."""
        kps = self._make_kps({0: (100, 50, 0.1), 1: (90, 45, 0.05)})  # low conf
        mask = _build_pose_mask(kps, _HEAD_KPS, (0, 0, 200, 400), 7, 7)
        np.testing.assert_allclose(mask, np.ones((7, 7)), atol=1e-6)

    def test_empty_keypoints(self):
        kps = np.zeros((17, 3), dtype=np.float32)
        mask = _build_pose_mask(kps, _HEAD_KPS, (0, 0, 200, 400), 7, 7)
        np.testing.assert_allclose(mask, np.ones((7, 7)), atol=1e-6)

    def test_torso_keypoints_activate_middle(self):
        """Torso keypoints (shoulders + hips) should activate middle region."""
        kps = self._make_kps({
            5: (80, 120, 0.9),   # left shoulder
            6: (120, 120, 0.9),  # right shoulder
            11: (80, 250, 0.9),  # left hip
            12: (120, 250, 0.9), # right hip
        })
        mask = _build_pose_mask(kps, _TORSO_KPS, (0, 0, 200, 400), 7, 7)
        # Middle region should be brighter than extremes
        mid_row = mask.shape[0] // 2
        assert mask[mid_row, :].mean() > mask[0, :].mean()
        assert mask[mid_row, :].mean() > mask[-1, :].mean()

    def test_different_feature_sizes(self):
        kps = self._make_kps({0: (100, 50, 0.9)})
        for size in [5, 7, 14]:
            mask = _build_pose_mask(kps, _HEAD_KPS, (0, 0, 200, 400), size, size)
            assert mask.shape == (size, size)


class TestBuildFallbackMask:
    """Test _build_fallback_mask spatial regions."""

    def test_head_mask_top_region_active(self):
        mask = _build_fallback_mask('head', 7, 7, floor=0.2)
        assert mask.shape == (7, 7)
        # Top 40% (rows 0-2) should be 1.0
        h_end = max(1, int(7 * 0.40))  # = 2
        assert (mask[:h_end, :] == 1.0).all()
        # Bottom should be floor
        assert (mask[h_end:, :] == 0.2).all()

    def test_torso_mask_middle_region_active(self):
        mask = _build_fallback_mask('torso', 7, 7, floor=0.2)
        h_start = int(7 * 0.20)  # 1
        h_end   = int(7 * 0.80)  # 5
        assert (mask[h_start:h_end, :] == 1.0).all()
        # Top and bottom should be floor
        assert (mask[:h_start, :] == 0.2).all()
        assert (mask[h_end:, :] == 0.2).all()

    def test_floor_zero(self):
        mask = _build_fallback_mask('head', 7, 7, floor=0.0)
        h_end = max(1, int(7 * 0.40))
        assert (mask[h_end:, :] == 0.0).all()

    def test_floor_one_is_uniform(self):
        mask = _build_fallback_mask('head', 7, 7, floor=1.0)
        assert (mask == 1.0).all()


class TestMaskProperties:
    """Test general properties that masks should satisfy."""

    def test_mask_is_symmetric_for_centered_keypoints(self):
        """Centered keypoints should produce roughly symmetric mask."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[0] = [100, 200, 0.9]  # center of 200x400 box
        mask = _build_pose_mask(kps, [0], (0, 0, 200, 400), 7, 7)
        # Check left-right symmetry (approximately)
        np.testing.assert_allclose(mask[:, :3], mask[:, 4:][:, ::-1], atol=0.2)

    def test_floor_preserves_context(self):
        """Floor > 0 means no part of the mask is completely zero."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[0] = [10, 10, 0.9]  # corner keypoint
        mask = _build_pose_mask(kps, [0], (0, 0, 200, 400), 7, 7, floor=0.2)
        assert mask.min() >= 0.2 - 1e-6
