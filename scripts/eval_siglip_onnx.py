"""
Evaluate SigLIP ONNX INT8 drift vs PyTorch baseline on real violation crops.

Uses the same head/torso split + preprocess as production inference.
Crops from --crops-dir are treated as full-person bounding boxes.

Usage:
    python scripts/eval_siglip_onnx.py
    python scripts/eval_siglip_onnx.py --crops-dir violation_crops --onnx model_cache/siglip_image_encoder.onnx
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import numpy as np
import torch
import open_clip
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    SIGLIP_MODEL_NAME, SIGLIP_PRETRAINED, SIGLIP_PROMPTS,
    SIGLIP_ONNX_PATH, PPE_THRESHOLDS,
)
from src.inference.classifier import _classify_siglip_batch


def _load_pytorch_bundle(device: torch.device) -> dict:
    print(f"Loading PyTorch model ({SIGLIP_MODEL_NAME})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        SIGLIP_MODEL_NAME, pretrained=SIGLIP_PRETRAINED,
    )
    model.eval().to(device)

    tokenizer = open_clip.get_tokenizer(SIGLIP_MODEL_NAME)
    text_features: dict = {}
    with torch.no_grad():
        for category, prompts in SIGLIP_PROMPTS.items():
            pos_tok = tokenizer(prompts["positive"]).to(device)
            neg_tok = tokenizer(prompts["negative"]).to(device)
            pos_feat = model.encode_text(pos_tok).mean(dim=0)
            neg_feat = model.encode_text(neg_tok).mean(dim=0)
            pos_feat = pos_feat / pos_feat.norm()
            neg_feat = neg_feat / neg_feat.norm()
            text_features[category] = torch.stack([pos_feat, neg_feat])

    return {
        "model": model,
        "preprocess": preprocess,
        "text_features": text_features,
        "logit_scale": float(model.logit_scale.exp().item()),
    }


def _load_onnx_bundle(onnx_path: str, pt_bundle: dict) -> dict:
    print(f"Loading ONNX session ({onnx_path})...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return {
        "model": session,
        "preprocess": pt_bundle["preprocess"],
        "text_features": pt_bundle["text_features"],
        "format": "onnx",
        "logit_scale": pt_bundle["logit_scale"],
    }


def _ppe_decision(prob: float, category: str) -> str:
    thr = PPE_THRESHOLDS[category]
    if prob >= thr["ok"]:
        return "ok"
    if prob >= thr["warn"]:
        return "warn"
    return "violation"


def _run_eval(crops_dir: str, onnx_path: str) -> None:
    device = torch.device("cpu")
    pt_bundle = _load_pytorch_bundle(device)
    onnx_bundle = _load_onnx_bundle(onnx_path, pt_bundle)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    crop_paths = [
        p for p in Path(crops_dir).iterdir()
        if p.suffix.lower() in image_exts
    ]
    if not crop_paths:
        print(f"No images found in {crops_dir}")
        return

    print(f"Evaluating {len(crop_paths)} crops...\n")

    hardhat_diffs: list[float] = []
    vest_diffs: list[float] = []
    hardhat_flips = 0
    vest_flips = 0
    failed = 0

    for path in sorted(crop_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            failed += 1
            continue

        h, w = frame.shape[:2]
        box = [[0, 0, w, h]]

        pt_results = _classify_siglip_batch(pt_bundle, frame, box, device)
        onnx_results = _classify_siglip_batch(onnx_bundle, frame, box, device)

        if pt_results[0] is None or onnx_results[0] is None:
            failed += 1
            continue

        pt_probs = pt_results[0]["probs"]
        onnx_probs = onnx_results[0]["probs"]

        hh_diff = abs(float(pt_probs[0]) - float(onnx_probs[0]))
        ve_diff = abs(float(pt_probs[1]) - float(onnx_probs[1]))
        hardhat_diffs.append(hh_diff)
        vest_diffs.append(ve_diff)

        if _ppe_decision(float(pt_probs[0]), "hardhat") != _ppe_decision(float(onnx_probs[0]), "hardhat"):
            hardhat_flips += 1
        if _ppe_decision(float(pt_probs[1]), "vest") != _ppe_decision(float(onnx_probs[1]), "vest"):
            vest_flips += 1

    n = len(hardhat_diffs)
    if n == 0:
        print("No valid crops processed.")
        return

    def _stats(diffs: list[float]) -> str:
        arr = np.array(diffs)
        return (
            f"mean={arr.mean():.4f}  max={arr.max():.4f}  "
            f"p95={np.percentile(arr, 95):.4f}  p99={np.percentile(arr, 99):.4f}"
        )

    print("=" * 60)
    print(f"Crops evaluated : {n}  (failed/skipped: {failed})")
    print()
    print(f"Hardhat probability drift  |pt - onnx|")
    print(f"  {_stats(hardhat_diffs)}")
    print(f"  Decision flips (ok/warn/violation): {hardhat_flips} / {n}  ({100*hardhat_flips/n:.1f}%)")
    print()
    print(f"Vest probability drift  |pt - onnx|")
    print(f"  {_stats(vest_diffs)}")
    print(f"  Decision flips (ok/warn/violation): {vest_flips} / {n}  ({100*vest_flips/n:.1f}%)")
    print("=" * 60)

    total_flips = hardhat_flips + vest_flips
    total_decisions = 2 * n
    flip_rate = total_flips / total_decisions
    verdict = "PASS" if flip_rate <= 0.05 else "REVIEW"
    print(f"\nOverall flip rate: {flip_rate:.1%}  [{verdict}]")
    print("(PASS = <=5% decisions change between PyTorch and ONNX)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", default="violation_crops")
    parser.add_argument("--onnx", default=SIGLIP_ONNX_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ONNX file not found: {args.onnx}")
        print("Run: python scripts/export_siglip_onnx.py")
        sys.exit(1)

    _run_eval(args.crops_dir, args.onnx)


if __name__ == "__main__":
    main()
