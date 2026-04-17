"""
PPE Classifier - SigLIP Zero-Shot.

Replaces EfficientNetV2-B0 with SigLIP zero-shot classification.
Full-body crop -> SigLIP image embedding -> cosine similarity with text prompts
→ softmax → P(hardhat), P(vest).
"""

import cv2
import torch
import numpy as np
from PIL import Image
from src.config import (
    ROUTER_MIN_SIDE_PX,
    SIGLIP_USE_POSE,
    HARDHAT_VIS_MIN_SIDE,
    HARDHAT_VIS_MIN_BRIGHT,
    HARDHAT_VIS_MIN_SHARP,
    WHITE_HARDHAT_PRIOR_ENABLE,
    WHITE_HARDHAT_SAT_MAX,
    WHITE_HARDHAT_VAL_MIN,
    WHITE_HARDHAT_MIN_RATIO,
    WHITE_HARDHAT_PROB_BOOST,
    BRIGHT_HARDHAT_PRIOR_ENABLE,
    BRIGHT_HARDHAT_MIN_RATIO,
    BRIGHT_HARDHAT_PROB_BOOST,
    YELLOW_HARDHAT_H_MIN,
    YELLOW_HARDHAT_H_MAX,
    YELLOW_HARDHAT_S_MIN,
    YELLOW_HARDHAT_V_MIN,
    DARK_HEAD_PRIOR_ENABLE,
    DARK_HEAD_VAL_MAX,
    DARK_HEAD_MIN_RATIO,
    DARK_HEAD_PROB_PENALTY,
)


# Global flag — toggled via /api/cam_mode endpoint (kept for API compat, no-op)
cam_mode_enabled: bool = False
router_min_side_px: float = float(ROUTER_MIN_SIDE_PX)
siglip_use_pose: bool = bool(SIGLIP_USE_POSE)

_KP_CONF_THR = 0.30
_HEAD_KPS = [0, 1, 2, 3, 4]
_TORSO_KPS = [5, 6, 11, 12]


def has_effnet_model(model_stage2) -> bool:
    if not isinstance(model_stage2, dict):
        return False
    effnet_bundle = model_stage2.get("effnet")
    return isinstance(effnet_bundle, dict) and effnet_bundle.get("model") is not None


def _get_siglip_bundle(model_stage2):
    # Backward compatibility: accept either plain SigLIP bundle or legacy CLIP key.
    if isinstance(model_stage2, dict) and "siglip" in model_stage2:
        return model_stage2["siglip"]
    if isinstance(model_stage2, dict) and "clip" in model_stage2:
        return model_stage2["clip"]
    return model_stage2


def _compute_router_indices(model_stage2, boxes_np: np.ndarray):
    widths = boxes_np[:, 2] - boxes_np[:, 0]
    heights = boxes_np[:, 3] - boxes_np[:, 1]
    min_sides = np.minimum(widths, heights)

    use_effnet_mask = (min_sides < float(router_min_side_px))
    if not has_effnet_model(model_stage2):
        use_effnet_mask = np.zeros_like(use_effnet_mask, dtype=bool)

    effnet_idx = np.where(use_effnet_mask)[0].tolist()
    siglip_idx = np.where(~use_effnet_mask)[0].tolist()
    return effnet_idx, siglip_idx


def should_run_pose_for_boxes(model_stage2, boxes) -> bool:
    if not siglip_use_pose or boxes is None or len(boxes) == 0:
        return False
    boxes_np = np.asarray(boxes, dtype=np.float32)
    _, siglip_idx = _compute_router_indices(model_stage2, boxes_np)
    return len(siglip_idx) > 0


def _match_pose_to_box(box, pose_results):
    if pose_results is None:
        return None

    bx1, by1, bx2, by2 = map(float, box)
    candidates = []

    for r in pose_results:
        if getattr(r, "keypoints", None) is None:
            continue
        for person_kps in r.keypoints.data:
            kps = person_kps.cpu().numpy()
            valid = kps[kps[:, 2] >= _KP_CONF_THR]
            if len(valid) == 0:
                continue

            min_x, min_y = valid[:, 0].min(), valid[:, 1].min()
            max_x, max_y = valid[:, 0].max(), valid[:, 1].max()
            inter_x1 = max(bx1, min_x)
            inter_y1 = max(by1, min_y)
            inter_x2 = min(bx2, max_x)
            inter_y2 = min(by2, max_y)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                continue

            kps_area = max(1.0, (max_x - min_x) * (max_y - min_y))
            overlap = inter_area / kps_area
            cx = (min_x + max_x) * 0.5
            cy = (min_y + max_y) * 0.5
            bx = (bx1 + bx2) * 0.5
            by = (by1 + by2) * 0.5
            dist = np.hypot(cx - bx, cy - by)
            candidates.append((overlap, -dist, kps))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_overlap = candidates[0][0]
    if best_overlap < 0.15:
        return None
    # Ambiguous associations in crowded scenes: fall back to bbox region crop.
    if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 0.07:
        return None
    return candidates[0][2]


def _crop_from_kps(frame, kps, kp_indices, fallback_box, pad_ratio=0.40):
    img_h, img_w = frame.shape[:2]
    valid = [
        kps[idx] for idx in kp_indices
        if idx < len(kps) and kps[idx][2] >= _KP_CONF_THR
    ]
    if len(valid) < 2:
        return None, None

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half = max((max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0, 12.0) * (1.0 + pad_ratio)

    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    return crop, (x1, y1, x2, y2)


def _fallback_regions(frame, box):
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pw = int(bw * 0.10)
    ph = int(bh * 0.08)
    x1p = max(0, x1 - pw)
    x2p = min(img_w, x2 + pw)
    y1p = max(0, y1 - ph)
    y2p = min(img_h, y2 + ph)

    h_total = max(1, y2p - y1p)
    head_y2 = y1p + int(h_total * 0.42)
    torso_y1 = y1p + int(h_total * 0.20)
    torso_y2 = y1p + int(h_total * 0.85)

    head = frame[y1p:head_y2, x1p:x2p]
    torso = frame[torso_y1:torso_y2, x1p:x2p]
    if head.size == 0:
        head = frame[y1p:y2p, x1p:x2p]
    if torso.size == 0:
        torso = frame[y1p:y2p, x1p:x2p]
    return head, (x1p, y1p, x2p, head_y2), torso, (x1p, torso_y1, x2p, torso_y2)


def _ensure_min_crop(crop, min_w=80, min_h=80):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if w < min_w or h < min_h:
        scale = max(min_w / max(w, 1), min_h / max(h, 1))
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    return crop


def _estimate_head_quality(head_crop):
    if head_crop is None or head_crop.size == 0:
        return {
            "hardhat_visibility_low": True,
            "head_min_side": 0.0,
            "head_brightness": 0.0,
            "head_sharpness": 0.0,
        }
    h, w = head_crop.shape[:2]
    min_side = float(min(h, w))
    gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[:, :, 2].mean())

    visibility_low = (
        min_side < float(HARDHAT_VIS_MIN_SIDE)
        or sharpness < float(HARDHAT_VIS_MIN_SHARP)
        or brightness < float(HARDHAT_VIS_MIN_BRIGHT)
    )
    return {
        "hardhat_visibility_low": bool(visibility_low),
        "head_min_side": min_side,
        "head_brightness": brightness,
        "head_sharpness": sharpness,
    }


def _estimate_white_hardhat_ratio(head_crop):
    if head_crop is None or head_crop.size == 0:
        return 0.0

    h, w = head_crop.shape[:2]
    top_h = max(1, int(h * 0.60))
    roi = head_crop[:top_h, :]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    white_mask = (sat <= float(WHITE_HARDHAT_SAT_MAX)) & (val >= float(WHITE_HARDHAT_VAL_MIN))
    return float(white_mask.mean())


def _estimate_bright_hardhat_ratio(head_crop):
    if head_crop is None or head_crop.size == 0:
        return 0.0

    h, w = head_crop.shape[:2]
    top_h = max(1, int(h * 0.60))
    roi = head_crop[:top_h, :]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    white_mask = (sat <= float(WHITE_HARDHAT_SAT_MAX)) & (val >= float(WHITE_HARDHAT_VAL_MIN))
    yellow_mask = (
        (hue >= float(YELLOW_HARDHAT_H_MIN))
        & (hue <= float(YELLOW_HARDHAT_H_MAX))
        & (sat >= float(YELLOW_HARDHAT_S_MIN))
        & (val >= float(YELLOW_HARDHAT_V_MIN))
    )
    bright_mask = white_mask | yellow_mask
    return float(bright_mask.mean())


def _estimate_dark_head_ratio(head_crop):
    if head_crop is None or head_crop.size == 0:
        return 0.0

    h, w = head_crop.shape[:2]
    top_h = max(1, int(h * 0.60))
    roi = head_crop[:top_h, :]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2].astype(np.float32)
    dark_mask = val <= float(DARK_HEAD_VAL_MAX)
    return float(dark_mask.mean())


def _draw_siglip_focus_overlay(frame, head_rect, torso_rect, visibility_low=False):
    if head_rect is not None:
        x1, y1, x2, y2 = head_rect
        color = (0, 165, 255) if visibility_low else (56, 139, 253)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        tag = "SIGLIP HEAD" if not visibility_low else "SIGLIP HEAD LOW-VIS"
        cv2.putText(frame, tag, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if torso_rect is not None:
        x1, y1, x2, y2 = torso_rect
        color = (80, 220, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, "SIGLIP TORSO", (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def _classify_siglip_batch(siglip_bundle, frame, boxes, device, pose_results=None):
    model = siglip_bundle["model"]
    preprocess = siglip_bundle["preprocess"]
    text_feats = siglip_bundle["text_features"]
    is_onnx = siglip_bundle.get("format") == "onnx"

    head_tensors = []
    torso_tensors = []
    valid_idx = []
    quality_meta = []
    focus_rects = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box)
        if (x2 - x1) < 10 or (y2 - y1) < 15:
            continue

        head_crop, head_rect = None, None
        torso_crop, torso_rect = None, None
        if siglip_use_pose and pose_results is not None:
            kps = _match_pose_to_box(box, pose_results)
            if kps is not None:
                head_crop, head_rect = _crop_from_kps(frame, kps, _HEAD_KPS, box, pad_ratio=0.45)
                torso_crop, torso_rect = _crop_from_kps(frame, kps, _TORSO_KPS, box, pad_ratio=0.25)

        if head_crop is None or torso_crop is None:
            fallback_head, fallback_head_rect, fallback_torso, fallback_torso_rect = _fallback_regions(frame, box)
            if head_crop is None:
                head_crop, head_rect = fallback_head, fallback_head_rect
            if torso_crop is None:
                torso_crop, torso_rect = fallback_torso, fallback_torso_rect

        head_quality = _estimate_head_quality(head_crop)

        head_crop = _ensure_min_crop(head_crop, min_w=72, min_h=72)
        torso_crop = _ensure_min_crop(torso_crop, min_w=96, min_h=96)
        if head_crop is None or torso_crop is None:
            continue

        head_pil = Image.fromarray(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB))
        torso_pil = Image.fromarray(cv2.cvtColor(torso_crop, cv2.COLOR_BGR2RGB))
        head_tensors.append(preprocess(head_pil))
        torso_tensors.append(preprocess(torso_pil))
        valid_idx.append(i)
        quality_meta.append(head_quality)
        focus_rects.append((head_rect, torso_rect))

    if not valid_idx:
        return [None] * len(boxes)

    head_batch = torch.stack(head_tensors).to(device)
    torso_batch = torch.stack(torso_tensors).to(device)

    with torch.no_grad():
        if is_onnx:
            _input_name = model.get_inputs()[0].name
            head_features = torch.from_numpy(
                model.run(None, {_input_name: head_batch.cpu().numpy().astype(np.float32)})[0]
            ).to(device)
            torso_features = torch.from_numpy(
                model.run(None, {_input_name: torso_batch.cpu().numpy().astype(np.float32)})[0]
            ).to(device)
            logit_scale = torch.tensor(siglip_bundle["logit_scale"]).to(device)
        else:
            head_features = model.encode_image(head_batch)
            torso_features = model.encode_image(torso_batch)
            logit_scale = model.logit_scale.exp()

        head_features = head_features / head_features.norm(dim=-1, keepdim=True)
        torso_features = torso_features / torso_features.norm(dim=-1, keepdim=True)

        h_sim = logit_scale * head_features @ text_feats["hardhat"].T
        h_prob = h_sim.softmax(dim=-1)[:, 0].cpu().numpy()
        v_sim = logit_scale * torso_features @ text_feats["vest"].T
        v_prob = v_sim.softmax(dim=-1)[:, 0].cpu().numpy()

    results = [None] * len(boxes)
    for j, idx in enumerate(valid_idx):
        quality = quality_meta[j]
        head_rect, torso_rect = focus_rects[j]
        if head_rect is not None:
            x1, y1, x2, y2 = map(int, head_rect)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            head_crop_for_prior = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        else:
            head_crop_for_prior = None

        white_ratio = _estimate_white_hardhat_ratio(head_crop_for_prior)
        bright_ratio = _estimate_bright_hardhat_ratio(head_crop_for_prior)
        dark_ratio = _estimate_dark_head_ratio(head_crop_for_prior)
        quality["white_hardhat_ratio"] = white_ratio
        quality["bright_hardhat_ratio"] = bright_ratio
        quality["dark_head_ratio"] = dark_ratio

        # Backward-compatible white prior.
        if WHITE_HARDHAT_PRIOR_ENABLE and white_ratio >= float(WHITE_HARDHAT_MIN_RATIO):
            if h_prob[j] < 0.90:
                h_prob[j] = min(0.95, h_prob[j] + float(WHITE_HARDHAT_PROB_BOOST))

        # New bright-head prior: reward bright helmet-like colors (white/yellow).
        if BRIGHT_HARDHAT_PRIOR_ENABLE and bright_ratio >= float(BRIGHT_HARDHAT_MIN_RATIO):
            if h_prob[j] < 0.90:
                h_prob[j] = min(0.95, h_prob[j] + float(BRIGHT_HARDHAT_PROB_BOOST))

        # New dark-head prior: penalize hardhat confidence when head region is mostly dark.
        if DARK_HEAD_PRIOR_ENABLE and dark_ratio >= float(DARK_HEAD_MIN_RATIO):
            h_prob[j] = max(0.02, h_prob[j] - float(DARK_HEAD_PROB_PENALTY))

        if cam_mode_enabled:
            _draw_siglip_focus_overlay(frame, head_rect, torso_rect, visibility_low=quality.get("hardhat_visibility_low", False))
        results[idx] = {
            "probs": np.array([float(h_prob[j]), float(v_prob[j])], dtype=np.float32),
            "router_model": "siglip",
            "quality": quality,
        }
    return results


def _classify_effnet_batch(effnet_bundle, frame, boxes, device):
    model = effnet_bundle["model"]
    preprocess = effnet_bundle["preprocess"]
    model_format = effnet_bundle.get("format", "pytorch")  # "pytorch" or "onnx"

    def _run_onnx(session, batch_np: np.ndarray) -> np.ndarray:
        input_meta = session.get_inputs()[0]
        input_shape = getattr(input_meta, "shape", None) or []
        max_batch = None
        if input_shape and len(input_shape) > 0 and isinstance(input_shape[0], int) and input_shape[0] > 0:
            max_batch = input_shape[0]

        if max_batch is None or batch_np.shape[0] <= max_batch:
            outputs = session.run(None, {input_meta.name: batch_np.astype(np.float32)})
            return np.array(outputs[0], dtype=np.float32)

        # Some exported ONNX models have a fixed batch dimension (often 1).
        # Run them in smaller chunks so inference does not crash on multi-box frames.
        collected = []
        for start in range(0, batch_np.shape[0], max_batch):
            chunk = batch_np[start:start + max_batch].astype(np.float32)
            outputs = session.run(None, {input_meta.name: chunk})
            collected.append(np.array(outputs[0], dtype=np.float32))
        return np.concatenate(collected, axis=0)

    img_h, img_w = frame.shape[:2]
    tensors = []
    valid_idx = []

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

        # Upscale tiny crops to preserve details before normalization.
        if crop_w < 80 or crop_h < 100:
            scale = max(80 / max(crop_w, 1), 100 / max(crop_h, 1))
            crop = cv2.resize(
                crop,
                (int(crop_w * scale), int(crop_h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(preprocess(pil))
        valid_idx.append(i)

    if not tensors:
        return [None] * len(boxes)

    if model_format == "onnx":
        # ONNX runtime expects numpy array
        batch_np = np.stack([t.numpy() for t in tensors], axis=0)
        logits = _run_onnx(model, batch_np)
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    else:
        # PyTorch format
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()

    results = [None] * len(boxes)
    for j, idx in enumerate(valid_idx):
        p = probs[j]
        if len(p) >= 2:
            out = np.array([float(p[0]), float(p[1])], dtype=np.float32)
        elif len(p) == 1:
            out = np.array([float(p[0]), float(p[0])], dtype=np.float32)
        else:
            out = np.array([0.0, 0.0], dtype=np.float32)
        results[idx] = {
            "probs": out,
            "router_model": "effnet",
            "quality": {"hardhat_visibility_low": False},
        }

    return results


def classify_ppe_batch(
    model_stage2, frame, boxes, device,
    tracker_ids=None,
    pose_results=None,
):
    """
    SigLIP zero-shot PPE classification on full-body crops.

    Args:
        siglip_bundle: dict with keys "model", "preprocess", "text_features"
        frame: BGR numpy array
        boxes: [N, 4] xyxy bounding boxes
        device: torch device
        tracker_ids: optional tracker IDs (unused by SigLIP, kept for interface compat)
        pose_results: optional pose results (unused by SigLIP, kept for interface compat)

    Returns:
        list of {"probs": np.array([hardhat_prob, vest_prob])} or None per box
    """
    if boxes is None or len(boxes) == 0:
        return []

    siglip_bundle = _get_siglip_bundle(model_stage2)
    effnet_bundle = model_stage2.get("effnet") if isinstance(model_stage2, dict) else None

    boxes_np = np.asarray(boxes, dtype=np.float32)
    effnet_idx, siglip_idx = _compute_router_indices(model_stage2, boxes_np)

    results = [None] * len(boxes)

    if effnet_idx:
        effnet_boxes = boxes_np[effnet_idx]
        effnet_results = _classify_effnet_batch(effnet_bundle, frame, effnet_boxes, device)
        for local_i, global_i in enumerate(effnet_idx):
            results[global_i] = effnet_results[local_i]

    if siglip_idx:
        siglip_boxes = boxes_np[siglip_idx]
        siglip_results = _classify_siglip_batch(siglip_bundle, frame, siglip_boxes, device, pose_results=pose_results)
        for local_i, global_i in enumerate(siglip_idx):
            results[global_i] = siglip_results[local_i]

    return results