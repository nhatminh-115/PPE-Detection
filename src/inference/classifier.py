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

# CAM EMA smoothing — reduces flicker
# {original_idx: {"hardhat": np, "vest": np}}
_cam_ema: dict = {}
CAM_EMA_ALPHA  = 0.1  # low = smoother, less flicker


def classify_ppe_batch(model_stage2, frame, boxes, device, tracker_ids=None):
    """Executes EfficientNetV2 inference, XAI Forward-CAM Color Penalization,
    and optional CAM heatmap overlay for hardhat + vest."""
    img_h, img_w = frame.shape[:2]
    tensors, valid_idx, raw_crops, crop_coords = [], [], [], []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        pw, ph = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
        x1c, y1c = max(0, x1 - pw), max(0, y1 - ph)
        x2c, y2c = min(img_w, x2 + pw), min(img_h, y2 + ph)

        crop_w, crop_h = x2c - x1c, y2c - y1c
        if crop_w < 10 or crop_h < 15:
            continue
        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue

        if crop_w < 80 or crop_h < 100:
            scale = max(80 / crop_w, 100 / crop_h)
            crop  = cv2.resize(
                crop,
                (int(crop_w * scale), int(crop_h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        raw_crops.append(crop)
        crop_coords.append((x1c, y1c, x2c, y2c))
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(val_transform(pil_img))
        valid_idx.append(i)

    if not tensors:
        return [None] * len(boxes)

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        features = model_stage2.forward_features(batch)
        pooled   = model_stage2.global_pool(features)
        logits   = model_stage2.classifier(pooled)
        probs    = torch.sigmoid(logits).cpu().numpy()

        # Compute CAM for both hardhat (idx 0) and vest (idx 1)
        weight_hardhat = model_stage2.classifier.weight[0]
        weight_vest    = model_stage2.classifier.weight[1]

        cam_hardhat = torch.relu(torch.einsum('bchw,c->bhw', features, weight_hardhat))
        cam_vest    = torch.relu(torch.einsum('bchw,c->bhw', features, weight_vest))

    for idx_in_batch, original_idx in enumerate(valid_idx):
        prob_hardhat = probs[idx_in_batch][0]
        crop_img     = raw_crops[idx_in_batch]
        h, w         = crop_img.shape[:2]

        # --- Hardhat CAM ---
        c_h     = cam_hardhat[idx_in_batch]
        c_h_max = c_h.max()
        if c_h_max > 1e-4:
            c_h = c_h / c_h_max
        c_h_np = c_h.cpu().numpy()

        # HSV penalization (hardhat only)
        if prob_hardhat > 0.4:
            c_resized = cv2.resize(c_h_np, (w, h))
            mask      = c_resized > 0.5
            if mask.sum() > 10:
                hsv_img     = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                mean_bright = hsv_img[:, :, 2][mask].mean()
                mean_sat    = hsv_img[:, :, 1][mask].mean()
                if mean_bright < 70:
                    probs[idx_in_batch][0] -= 0.55
                elif mean_sat < 80:
                    probs[idx_in_batch][0] -= 0.35
                probs[idx_in_batch][0] = max(0.0, probs[idx_in_batch][0])

        # --- Vest CAM ---
        c_v     = cam_vest[idx_in_batch]
        c_v_max = c_v.max()
        if c_v_max > 1e-4:
            c_v = c_v / c_v_max
        c_v_np = c_v.cpu().numpy()

        # --- CAM overlay with EMA smoothing ---
        if cam_mode_enabled:
            x1c, y1c, x2c, y2c = crop_coords[idx_in_batch]
            # Use tracker_id as EMA key for stability across frames
            key = int(tracker_ids[original_idx]) if tracker_ids is not None else original_idx

            if key not in _cam_ema:
                _cam_ema[key] = {"hardhat": c_h_np.copy(), "vest": c_v_np.copy()}
            else:
                _cam_ema[key]["hardhat"] = (
                    CAM_EMA_ALPHA * c_h_np + (1.0 - CAM_EMA_ALPHA) * _cam_ema[key]["hardhat"]
                )
                _cam_ema[key]["vest"] = (
                    CAM_EMA_ALPHA * c_v_np + (1.0 - CAM_EMA_ALPHA) * _cam_ema[key]["vest"]
                )

            # Render hardhat CAM (JET — blue to red)
            _render_cam_overlay(
                frame, _cam_ema[key]["hardhat"],
                x1c, y1c, x2c, y2c,
                colormap=cv2.COLORMAP_JET,
                border_color=(0, 165, 255),  # orange
                alpha=0.45,
            )
            # Render vest CAM (HOT — black to white via red/yellow)
            _render_cam_overlay(
                frame, _cam_ema[key]["vest"],
                x1c, y1c, x2c, y2c,
                colormap=cv2.COLORMAP_HOT,
                border_color=None,  # no second border
                alpha=0.30,         # lighter overlay, blends with hardhat
            )

    results = [None] * len(boxes)
    for idx_in_batch, original_idx in enumerate(valid_idx):
        results[original_idx] = {"probs": probs[idx_in_batch]}
    return results


def _render_cam_overlay(
    frame, cam_np, x1, y1, x2, y2,
    colormap=cv2.COLORMAP_JET,
    border_color=(0, 165, 255),
    alpha=0.45,
) -> None:
    """Renders heatmap overlay onto the frame region."""
    region_h = y2 - y1
    region_w = x2 - x1
    if region_h <= 0 or region_w <= 0:
        return

    cam_norm = cam_np.copy()
    if cam_norm.max() > 1e-4:
        cam_norm = cam_norm / cam_norm.max()

    cam_resized = cv2.resize(cam_norm, (region_w, region_h))
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap     = cv2.applyColorMap(cam_uint8, colormap)

    roi = frame[y1:y2, x1:x2]
    if roi.shape[:2] == heatmap.shape[:2]:
        blended = cv2.addWeighted(roi, 1.0 - alpha, heatmap, alpha, 0)
        frame[y1:y2, x1:x2] = blended

    if border_color:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)