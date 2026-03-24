import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.config import IMG_SIZE

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Global flag — toggled via /api/cam_mode endpoint
cam_mode_enabled: bool = False

# CAM EMA smoothing {tracker_id: {"hardhat": np, "vest": np}}
_cam_ema: dict = {}
CAM_EMA_ALPHA  = 0.1

# COCO keypoint indices
_HEAD_KPS    = [0, 1, 2, 3, 4]   # nose, eyes, ears
_TORSO_KPS   = [5, 6, 11, 12]    # shoulders, hips
_KP_CONF_THR = 0.3


# ---------------------------------------------------------------------------
# Pose-guided crop helpers
# ---------------------------------------------------------------------------

def _kp_crop(frame, keypoints, kp_indices, img_w, img_h, pad_ratio=0.4):
    """Extract crop region from keypoints. Returns (crop, x1, y1, x2, y2) or None."""
    valid_kps = [
        keypoints[i] for i in kp_indices
        if i < len(keypoints) and keypoints[i][2] >= _KP_CONF_THR
    ]
    if len(valid_kps) < 2:
        return None

    xs = [k[0] for k in valid_kps]
    ys = [k[1] for k in valid_kps]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    half = max((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2, 20) * (1 + pad_ratio)

    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, x1, y1, x2, y2


def _match_pose_to_box(box, pose_results, img_w, img_h):
    """Match person box to pose keypoints. Returns [17, 3] array or None."""
    if pose_results is None or len(pose_results) == 0:
        return None

    bx1, by1, bx2, by2 = map(int, box)

    for r in pose_results:
        if r.keypoints is None:
            continue
        for person_kps in r.keypoints.data:
            kps = person_kps.cpu().numpy()  # [17, 3]
            nose = kps[0]
            if nose[2] >= _KP_CONF_THR:
                cx, cy = nose[0], nose[1]
            else:
                valid = kps[kps[:, 2] >= _KP_CONF_THR]
                if len(valid) == 0:
                    continue
                cx, cy = valid[:, 0].mean(), valid[:, 1].mean()

            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                return kps

    return None


def _ensure_min_size(crop):
    h, w = crop.shape[:2]
    if w < 80 or h < 100:
        scale = max(80 / w, 100 / h)
        crop  = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return crop


# ---------------------------------------------------------------------------
# Main classify function
# ---------------------------------------------------------------------------

def classify_ppe_batch(
    model_stage2, frame, boxes, device,
    tracker_ids=None,
    pose_results=None,
):
    """
    EfficientNetV2 inference with pose-guided crop strategy.
    - Head crop (keypoints 0-4) → hardhat classification
    - Torso crop (keypoints 5,6,11,12) → vest classification
    - Fallback to % crop if pose unavailable
    """
    img_h, img_w = frame.shape[:2]
    crop_data  = []
    valid_idx  = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        pw, ph = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
        x1c, y1c = max(0, x1 - pw), max(0, y1 - ph)
        x2c, y2c = min(img_w, x2 + pw), min(img_h, y2 + ph)

        full_crop = frame[y1c:y2c, x1c:x2c]
        if full_crop.size == 0 or (x2c - x1c) < 10 or (y2c - y1c) < 15:
            continue

        head_result  = None
        torso_result = None

        if pose_results is not None:
            kps = _match_pose_to_box(box, pose_results, img_w, img_h)
            if kps is not None:
                head_result  = _kp_crop(frame, kps, _HEAD_KPS,  img_w, img_h, pad_ratio=0.5)
                torso_result = _kp_crop(frame, kps, _TORSO_KPS, img_w, img_h, pad_ratio=0.25)

        # Hardhat crop
        if head_result is not None:
            hc, hx1, hy1, hx2, hy2 = head_result
        else:
            hy2 = y1c + int((y2c - y1c) * 0.40)
            hc  = frame[y1c:hy2, x1c:x2c]
            hx1, hy1, hx2 = x1c, y1c, x2c
            if hc.size == 0:
                hc, hx1, hy1, hx2, hy2 = full_crop, x1c, y1c, x2c, y2c
        hc = _ensure_min_size(hc)

        # Vest crop
        if torso_result is not None:
            tc, tx1, ty1, tx2, ty2 = torso_result
        else:
            ty1 = y1c + int((y2c - y1c) * 0.20)
            ty2 = y1c + int((y2c - y1c) * 0.80)
            tc  = frame[ty1:ty2, x1c:x2c]
            tx1, tx2 = x1c, x2c
            if tc.size == 0:
                tc, tx1, ty1, tx2, ty2 = full_crop, x1c, y1c, x2c, y2c
        tc = _ensure_min_size(tc)

        crop_data.append({
            "hardhat_crop":   hc,
            "hardhat_coords": (hx1, hy1, hx2, hy2),
            "vest_crop":      tc,
            "vest_coords":    (tx1, ty1, tx2, ty2),
        })
        valid_idx.append(i)

    if not crop_data:
        return [None] * len(boxes)

    # Batch: [hardhat_0, vest_0, hardhat_1, vest_1, ...]
    tensors = []
    for cd in crop_data:
        for crop in [cd["hardhat_crop"], cd["vest_crop"]]:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensors.append(val_transform(pil))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        features = model_stage2.forward_features(batch)
        pooled   = model_stage2.global_pool(features)
        logits   = model_stage2.classifier(pooled)
        probs    = torch.sigmoid(logits).cpu().numpy()

        weight_hardhat = model_stage2.classifier.weight[0]
        weight_vest    = model_stage2.classifier.weight[1]
        cam_h = torch.relu(torch.einsum('bchw,c->bhw', features, weight_hardhat))
        cam_v = torch.relu(torch.einsum('bchw,c->bhw', features, weight_vest))

    results = [None] * len(boxes)

    for pair_idx, (original_idx, cd) in enumerate(zip(valid_idx, crop_data)):
        hi = pair_idx * 2
        vi = pair_idx * 2 + 1

        prob_hardhat = float(probs[hi][0])
        prob_vest    = float(probs[vi][1])

        # HSV penalization on head crop
        c_h_np = _normalize_cam(cam_h[hi])
        if prob_hardhat > 0.4:
            hc   = cd["hardhat_crop"]
            h, w = hc.shape[:2]
            mask = cv2.resize(c_h_np, (w, h)) > 0.5
            if mask.sum() > 10:
                hsv_img = cv2.cvtColor(hc, cv2.COLOR_BGR2HSV)
                if hsv_img[:, :, 2][mask].mean() < 70:
                    prob_hardhat -= 0.55
                elif hsv_img[:, :, 1][mask].mean() < 80:
                    prob_hardhat -= 0.35
                prob_hardhat = max(0.0, prob_hardhat)

        c_v_np = _normalize_cam(cam_v[vi])

        final_probs    = probs[hi].copy()
        final_probs[0] = prob_hardhat
        final_probs[1] = prob_vest

        # CAM overlay
        if cam_mode_enabled:
            key = int(tracker_ids[original_idx]) if tracker_ids is not None else original_idx
            hx1, hy1, hx2, hy2 = cd["hardhat_coords"]
            tx1, ty1, tx2, ty2 = cd["vest_coords"]

            if key not in _cam_ema:
                _cam_ema[key] = {"hardhat": c_h_np.copy(), "vest": c_v_np.copy()}
            else:
                _cam_ema[key]["hardhat"] = CAM_EMA_ALPHA * c_h_np + (1 - CAM_EMA_ALPHA) * _cam_ema[key]["hardhat"]
                _cam_ema[key]["vest"]    = CAM_EMA_ALPHA * c_v_np + (1 - CAM_EMA_ALPHA) * _cam_ema[key]["vest"]

            _render_cam_overlay(frame, _cam_ema[key]["hardhat"], hx1, hy1, hx2, hy2,
                                colormap=cv2.COLORMAP_JET, border_color=(0, 165, 255), alpha=0.45)
            _render_cam_overlay(frame, _cam_ema[key]["vest"], tx1, ty1, tx2, ty2,
                                colormap=cv2.COLORMAP_HOT, border_color=(0, 255, 200), alpha=0.35)

        results[original_idx] = {"probs": final_probs}

    return results


def _normalize_cam(cam_tensor):
    c = cam_tensor.clone()
    c_max = c.max()
    if c_max > 1e-4:
        c = c / c_max
    return c.cpu().numpy()


def _render_cam_overlay(frame, cam_np, x1, y1, x2, y2,
                        colormap=cv2.COLORMAP_JET,
                        border_color=(0, 165, 255),
                        alpha=0.45):
    rh, rw = y2 - y1, x2 - x1
    if rh <= 0 or rw <= 0:
        return
    cam_norm = cam_np.copy()
    if cam_norm.max() > 1e-4:
        cam_norm = cam_norm / cam_norm.max()
    heatmap = cv2.applyColorMap((cv2.resize(cam_norm, (rw, rh)) * 255).astype(np.uint8), colormap)
    roi = frame[y1:y2, x1:x2]
    if roi.shape[:2] == heatmap.shape[:2]:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, heatmap, alpha, 0)
    if border_color:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)