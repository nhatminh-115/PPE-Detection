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


def classify_ppe_batch(model_stage2, frame, boxes, device):
    """Executes EfficientNetV2 inference and XAI Forward-CAM Color Penalization."""
    img_h, img_w = frame.shape[:2]
    tensors, valid_idx, raw_crops = [], [], []

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

        weight_hardhat = model_stage2.classifier.weight[0]
        cam = torch.einsum('bchw,c->bhw', features, weight_hardhat)
        cam = torch.relu(cam)

    for idx_in_batch, original_idx in enumerate(valid_idx):
        prob_hardhat = probs[idx_in_batch][0]

        if prob_hardhat > 0.4:
            c = cam[idx_in_batch]
            c_max = c.max()
            if c_max > 1e-4:
                c = c / c_max

            c_np     = c.cpu().numpy()
            crop_img = raw_crops[idx_in_batch]
            h, w     = crop_img.shape[:2]
            c_resized = cv2.resize(c_np, (w, h))
            mask = c_resized > 0.5

            if mask.sum() > 10:
                hsv          = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                s_channel    = hsv[:, :, 1]
                v_channel    = hsv[:, :, 2]
                mean_sat     = s_channel[mask].mean()
                mean_bright  = v_channel[mask].mean()

                if mean_bright < 70:
                    probs[idx_in_batch][0] -= 0.55
                elif mean_sat < 80:
                    probs[idx_in_batch][0] -= 0.35

                probs[idx_in_batch][0] = max(0.0, probs[idx_in_batch][0])

    results = [None] * len(boxes)
    for idx_in_batch, original_idx in enumerate(valid_idx):
        results[original_idx] = {"probs": probs[idx_in_batch]}
    return results