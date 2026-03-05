import numpy as np
from src.config import WBF_IOU, SKIP_THR, MODEL_WEIGHTS

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False


def extract_boxes(results, img_w, img_h, class_filter=None):
    boxes, scores, labels = [], [], []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if class_filter is not None and cls not in class_filter:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
            scores.append(float(box.conf[0]))
            labels.append(cls)
    return boxes, scores, labels


def fuse_boxes(b1, s1, l1, b2, s2, l2, img_w, img_h):
    if not (b1 or b2):
        return np.empty((0, 4)), np.array([])

    boxes_list  = [b1, b2] if (b1 and b2) else ([b1] if b1 else [b2])
    scores_list = [s1, s2] if (b1 and b2) else ([s1] if b1 else [s2])
    labels_list = (
        [[float(x) for x in l1], [float(x) for x in l2]]
        if (b1 and b2)
        else [[float(x) for x in (l1 if b1 else l2)]]
    )
    w = MODEL_WEIGHTS if (b1 and b2) else [1]

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=w, iou_thr=WBF_IOU, skip_box_thr=SKIP_THR,
    )

    if len(fused_boxes) == 0:
        return np.empty((0, 4)), np.array([])

    fused_boxes = np.array(fused_boxes)
    fused_boxes[:, [0, 2]] *= img_w
    fused_boxes[:, [1, 3]] *= img_h
    return fused_boxes, np.array(fused_scores, dtype=float)