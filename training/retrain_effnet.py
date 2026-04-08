"""
Phase 4 — EfficientNet-B0 Retrain from Flywheel Data

Downloads confirmed human labels from S3 (or Supabase Storage as fallback),
builds a train/val dataset, fine-tunes EfficientNetV2-B0, and logs the new
weights to MLflow.

Run:
    python training/retrain_effnet.py

Environment:
  SUPABASE_URL, SUPABASE_SERVICE_KEY  -- label metadata source
  S3_BUCKET, S3_PREFIX                -- crop source (preferred)
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
  MIN_SAMPLES   -- abort if fewer confirmed labels (default 50)
  ROLLING_RATE  -- injected by CI for logging purposes
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import tempfile
from datetime import timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

MIN_SAMPLES  = int(os.environ.get("MIN_SAMPLES", "50"))
S3_BUCKET    = os.environ.get("S3_BUCKET", "")
S3_PREFIX    = os.environ.get("S3_PREFIX", "ppe-flywheel")
LABEL_COLS   = ["hardhat", "vest"]
TRAIN_SPLIT  = 0.85
IMG_SIZE     = 224
EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 5e-5
PATIENCE     = 7

MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "")
EXPERIMENT   = "PPE_Stage2_Classification_"
RUN_NAME     = "EfficientNetV2_B0_Flywheel"


# ── Dataset builder ────────────────────────────────────────────────────────────

def _fetch_all_confirmed_labels() -> list[dict]:
    """Return all confirmed (is_auto_labeled=False, skipped=False) labels."""
    from src.infrastructure.supabase import get_supabase_service_client
    client = get_supabase_service_client()
    if not client:
        raise RuntimeError("Supabase service client unavailable")

    rows = (
        client.table("ppe_labels")
        .select("merge_key, human_missing, crop_url, crop_sha256, crop_s3_key")
        .eq("is_auto_labeled", False)
        .eq("skipped", False)
        .not_.is_("crop_url", "null")
        .execute()
        .data or []
    )
    return rows


def _download_from_s3(sha256: str) -> bytes | None:
    if not S3_BUCKET or not sha256:
        return None
    import boto3
    from botocore.exceptions import ClientError
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    # Search for the sha256 key across any date partition
    prefix = f"{S3_PREFIX}/crops/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(f"/{sha256}.jpg"):
                resp = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                return resp["Body"].read()
    return None


def _download_from_url(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("retrain: crop download failed (%s): %s", url, exc)
        return None


def _row_to_label(row: dict) -> dict[str, int]:
    """Convert human_missing list to {hardhat: 0/1, vest: 0/1} where 1=wearing."""
    missing = row.get("human_missing") or []
    return {
        "hardhat": 0 if "hardhat" in missing else 1,
        "vest":    0 if "vest" in missing else 1,
    }


def build_dataset(work_dir: Path) -> tuple[Path, Path, Path]:
    """
    Download crops and build train/val CSV splits under work_dir.
    Returns (images_dir, train_csv_path, val_csv_path).
    """
    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = _fetch_all_confirmed_labels()
    logger.info("retrain: %d confirmed label rows fetched", len(rows))

    valid_rows: list[dict] = []
    for row in rows:
        sha256   = row.get("crop_sha256")
        crop_url = row.get("crop_url")
        img_name = f"{sha256 or row['merge_key'].replace('/', '_')}.jpg"
        img_path = images_dir / img_name

        if img_path.exists():
            valid_rows.append({**row, "_img_name": img_name})
            continue

        # Try S3 first, then Supabase Storage URL
        data = _download_from_s3(sha256) if sha256 else None
        if data is None:
            data = _download_from_url(crop_url) if crop_url else None

        if data:
            img_path.write_bytes(data)
            valid_rows.append({**row, "_img_name": img_name})
        else:
            logger.debug("retrain: could not download crop for %s", row.get("merge_key"))

    logger.info("retrain: %d valid crops with images", len(valid_rows))

    if len(valid_rows) < MIN_SAMPLES:
        raise RuntimeError(
            f"retrain: only {len(valid_rows)} valid samples, need >= {MIN_SAMPLES}"
        )

    random.shuffle(valid_rows)
    split_idx = int(len(valid_rows) * TRAIN_SPLIT)
    train_rows = valid_rows[:split_idx]
    val_rows   = valid_rows[split_idx:]

    def write_csv(path: Path, data: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_name"] + LABEL_COLS)
            writer.writeheader()
            for row in data:
                lbl = _row_to_label(row)
                writer.writerow({"image_name": row["_img_name"], **lbl})

    train_csv = work_dir / "train.csv"
    val_csv   = work_dir / "val.csv"
    write_csv(train_csv, train_rows)
    write_csv(val_csv, val_rows)

    logger.info("retrain: train=%d val=%d", len(train_rows), len(val_rows))
    return images_dir, train_csv, val_csv


# ── Training loop ──────────────────────────────────────────────────────────────

def train(images_dir: Path, train_csv: Path, val_csv: Path, out_dir: Path) -> str:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    # Local dataset class that works with the 2-column CSV (no mask needed)
    import pandas as pd
    from PIL import Image
    from torch.utils.data import Dataset

    class FlyWheelDataset(Dataset):
        def __init__(self, csv_path: Path, img_dir: Path, transform=None):
            self.df        = pd.read_csv(csv_path)
            self.img_dir   = img_dir
            self.transform = transform

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx):
            row      = self.df.iloc[idx]
            img_path = self.img_dir / row["image_name"]
            image    = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            labels = torch.tensor(
                [float(row[c]) for c in LABEL_COLS], dtype=torch.float32
            )
            return image, labels

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FlyWheelDataset(train_csv, images_dir, train_tf)
    val_ds   = FlyWheelDataset(val_csv,   images_dir, val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("retrain: device=%s train=%d val=%d", device, len(train_ds), len(val_ds))

    model = timm.create_model(
        "tf_efficientnetv2_b0", pretrained=False, num_classes=len(LABEL_COLS)
    ).to(device)

    # Load production weights as starting point if available
    prod_weights = Path("best_stage2_effnet.pt")
    if prod_weights.exists():
        try:
            state = torch.load(prod_weights, map_location=device)
            # Strip last layer if shape mismatches (old model has 4 outputs)
            if list(state.keys())[-1] in state:
                old_out = state[list(state.keys())[-1]].shape[0]
                if old_out != len(LABEL_COLS):
                    for k in list(state.keys()):
                        if "classifier" in k:
                            del state[k]
                    model.load_state_dict(state, strict=False)
                    logger.info("retrain: loaded prod weights (partial, head replaced)")
                else:
                    model.load_state_dict(state)
                    logger.info("retrain: loaded prod weights")
        except Exception as exc:
            logger.warning("retrain: could not load prod weights (%s), training from scratch", exc)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss  = float("inf")
    early_stop_n   = 0
    best_path      = out_dir / "best_flywheel_effnet.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_loss += criterion(model(images), labels).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        logger.info("Epoch %d | train=%.4f val=%.4f", epoch + 1, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_path)
            early_stop_n = 0
        else:
            early_stop_n += 1
            if early_stop_n >= PATIENCE:
                logger.info("retrain: early stop at epoch %d", epoch + 1)
                break

    logger.info("retrain: done. best val_loss=%.4f", best_val_loss)
    return str(best_path)


# ── ONNX Export ────────────────────────────────────────────────────────────────

def export_to_onnx(pt_path: str, onnx_path: str) -> bool:
    """
    Export PyTorch EfficientNetV2-B0 weights to ONNX format.
    
    Args:
        pt_path: Path to .pt weights file
        onnx_path: Path to save ONNX model
        
    Returns:
        True if successful, False otherwise
    """
    import torch
    import timm
    
    try:
        logger.info(f"Exporting PyTorch model to ONNX: {pt_path} -> {onnx_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and state dict
        model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=len(LABEL_COLS))
        state = torch.load(pt_path, map_location=device)
        
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        
        model.load_state_dict(state, strict=False)
        model.eval().to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["images"],
            output_names=["output"],
            opset_version=14,
            do_constant_folding=True,
            dynamic_axes={"images": {0: "batch_size"}},
            verbose=False,
        )
        
        logger.info(f"✓ ONNX export successful: {onnx_path}")
        
        # Verify ONNX model is readable
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model validation passed")
            return True
        except ImportError:
            logger.warning("onnx package not installed, skipping validation")
            return True
        except Exception as ex:
            logger.warning(f"ONNX validation failed: {ex}")
            return False
            
    except Exception as ex:
        logger.error(f"ONNX export failed: {ex}")
        return False


# ── MLflow logging ─────────────────────────────────────────────────────────────

def log_to_mlflow(weights_path: str, n_train: int, n_val: int) -> str:
    """
    Log training artifacts to MLflow.
    
    Logs both PyTorch (.pt) and ONNX formats:
    - .pt for retrain iterations (prefer latest good run)
    - .onnx for production inference (6x faster)
    """
    import mlflow
    from pathlib import Path

    if MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)

    mlflow.set_experiment(EXPERIMENT)

    rolling_rate = os.environ.get("ROLLING_RATE", "unknown")

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params({
            "model":          "tf_efficientnetv2_b0",
            "label_cols":     str(LABEL_COLS),
            "epochs":         EPOCHS,
            "batch_size":     BATCH_SIZE,
            "lr":             LR,
            "train_samples":  n_train,
            "val_samples":    n_val,
            "trigger":        "drift",
            "rolling_rate":   rolling_rate,
        })
        
        # Log PyTorch weights
        mlflow.log_artifact(weights_path, artifact_path="weights")
        logger.info(f"Logged PyTorch artifact: {weights_path}")
        
        # Export and log ONNX model
        onnx_path = str(Path(weights_path).parent / "effnet_ppe.onnx")
        if export_to_onnx(weights_path, onnx_path):
            mlflow.log_artifact(onnx_path, artifact_path="weights")
            logger.info(f"Logged ONNX artifact: {onnx_path}")
            mlflow.log_param("export_onnx", "success")
        else:
            logger.warning("ONNX export failed, only PyTorch artifact logged")
            mlflow.log_param("export_onnx", "failed")
        
        run_id = run.info.run_id

    logger.info("retrain: logged to MLflow run_id=%s", run_id)
    return run_id


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ppe_retrain_") as tmp:
        work_dir = Path(tmp)
        images_dir, train_csv, val_csv = build_dataset(work_dir)

        n_train = sum(1 for _ in open(train_csv)) - 1
        n_val   = sum(1 for _ in open(val_csv)) - 1

        out_dir      = work_dir / "output"
        weights_path = train(images_dir, train_csv, val_csv, out_dir)
        run_id       = log_to_mlflow(weights_path, n_train, n_val)

        print(f"[retrain] Complete. MLflow run_id={run_id}")
        print(f"[retrain] Update EFFNET_RUN_ID in src/config.py to: {run_id}")


if __name__ == "__main__":
    main()
