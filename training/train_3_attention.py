"""
Train Stage 2 — EfficientNetV2-B0 with Pose-Guided Soft Attention.

This script fine-tunes the classifier head of a pre-trained EfficientNetV2-B0
to work with masked pooling (head mask for hardhat, torso mask for vest).

Usage:
    # Step 1: Pre-compute keypoints (run once)
    python train_3_attention.py --precompute

    # Step 2: Fine-tune classifier head
    python train_3_attention.py --train
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
import torchmetrics
from tqdm import tqdm
from dotenv import load_dotenv
from ppe_dataset_v2 import PPEAttentionDataset, get_val_transforms

load_dotenv()

# ==============================================================================
# CONFIG
# ==============================================================================
DAGSHUB_URI     = "https://dagshub.com/nhatminh-115/PPE-Detection.mlflow"
EXPERIMENT_NAME = "PPE_Stage2_Classification_"
RUN_NAME        = "EfficientNetV2_B0_Attention_1"

CSV_TRAIN = r"datasets\stage2_ultralytics\train\labels.csv"
IMG_TRAIN = r"datasets\stage2_ultralytics\train\images"
CSV_VAL   = r"datasets\stage2_ultralytics\val\labels.csv"
IMG_VAL   = r"datasets\stage2_ultralytics\val\images"

# Keypoints cache paths
KP_CACHE_TRAIN = r"datasets\stage2_ultralytics\train\keypoints.npz"
KP_CACHE_VAL   = r"datasets\stage2_ultralytics\val\keypoints.npz"

POSE_MODEL_PATH = "yolo26m-pose.pt"

# Only fine-tune for hardhat (idx 0) and vest (idx 1)
# Dataset still has 4 classes for compatibility, but loss is masked
LABEL_COLS  = ['hardhat', 'vest', 'gloves', 'goggles']
NUM_CLASSES = 4
IMG_SIZE    = 224
FEAT_SIZE   = 7     # EffNetV2-B0 feature map spatial size at 224×224

# Training params — short fine-tune (only classifier head)
EPOCHS     = 20
BATCH_SIZE = 32
LR         = 5e-4    # slightly higher than full training since fewer params
PATIENCE   = 8

# Attention params (match config.py)
ATTENTION_SIGMA = 0.4
ATTENTION_FLOOR = 0.2

# Label counts for pos_weight (from original dataset)
TOTAL_TRAIN = 1755
POS_COUNTS  = [1202, 1209, 628, 449]  # hardhat, vest, gloves, goggles

# Source weights — pre-trained from train_2.py
SOURCE_WEIGHTS_RUN_ID = "e44391ef94b54ce3b867d46b7ce33ec3"


# ==============================================================================
# AUGMENTATION
# ==============================================================================
def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    ])


# ==============================================================================
# PRE-COMPUTE KEYPOINTS
# ==============================================================================
def precompute():
    """Run YOLO-pose on train and val images, save keypoints to .npz cache."""
    for csv_path, img_dir, output_path, split in [
        (CSV_TRAIN, IMG_TRAIN, KP_CACHE_TRAIN, "train"),
        (CSV_VAL,   IMG_VAL,   KP_CACHE_VAL,   "val"),
    ]:
        if os.path.exists(output_path):
            print(f"[SKIP] {split} keypoints cache already exists: {output_path}")
            continue
        print(f"\n[PRE-COMPUTE] {split}...")
        PPEAttentionDataset.precompute_keypoints(
            img_dir=img_dir,
            csv_path=csv_path,
            output_path=output_path,
            pose_model_path=POSE_MODEL_PATH,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            batch_size=16,
        )
    print("\n[DONE] Keypoints pre-computed for train and val splits.")


# ==============================================================================
# MASKED FORWARD PASS — matching inference logic
# ==============================================================================
def masked_forward(model, images, head_masks, torso_masks):
    """
    Forward pass with pose-guided masked pooling.
    Returns logits [B, 2] for [hardhat, vest].
    """
    features = model.forward_features(images)           # [B, C, Hf, Wf]

    weight = model.classifier.weight                    # [num_classes, C]
    bias   = model.classifier.bias                      # [num_classes]

    # Hardhat (class 0): masked by head
    head_feat    = features * head_masks                # [B, C, Hf, Wf]
    head_pooled  = head_feat.sum(dim=[2, 3]) / head_masks.sum(dim=[2, 3]).clamp(min=1e-4)
    head_logit   = (head_pooled * weight[0]).sum(dim=1, keepdim=True) + bias[0]  # [B, 1]

    # Vest (class 1): masked by torso
    torso_feat   = features * torso_masks
    torso_pooled = torso_feat.sum(dim=[2, 3]) / torso_masks.sum(dim=[2, 3]).clamp(min=1e-4)
    torso_logit  = (torso_pooled * weight[1]).sum(dim=1, keepdim=True) + bias[1]  # [B, 1]

    return torch.cat([head_logit, torso_logit], dim=1)  # [B, 2]


# ==============================================================================
# TRAIN
# ==============================================================================
def train():
    # Check keypoints cache
    for cache_path, split in [(KP_CACHE_TRAIN, "train"), (KP_CACHE_VAL, "val")]:
        if not os.path.exists(cache_path):
            print(f"[ERROR] Keypoints cache not found: {cache_path}")
            print("Run: python train_3_attention.py --precompute")
            return

    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Device: {device}")

    # Datasets with keypoints cache
    train_ds = PPEAttentionDataset(
        CSV_TRAIN, IMG_TRAIN, get_train_transforms(IMG_SIZE), LABEL_COLS,
        keypoints_cache=KP_CACHE_TRAIN, feat_size=FEAT_SIZE,
        sigma=ATTENTION_SIGMA, floor=ATTENTION_FLOOR,
    )
    val_ds = PPEAttentionDataset(
        CSV_VAL, IMG_VAL, get_val_transforms(IMG_SIZE), LABEL_COLS,
        keypoints_cache=KP_CACHE_VAL, feat_size=FEAT_SIZE,
        sigma=ATTENTION_SIGMA, floor=ATTENTION_FLOOR,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Load pre-trained model
    print("[SYSTEM] Loading pre-trained EfficientNetV2-B0...")
    model = timm.create_model('tf_efficientnetv2_b0', pretrained=False,
                              num_classes=NUM_CLASSES).to(device)

    # Try to load weights from MLflow or local
    weights_path = None
    try:
        from mlflow.artifacts import download_artifacts
        weights_path = download_artifacts(
            artifact_uri=f"runs:/{SOURCE_WEIGHTS_RUN_ID}/weights/best_stage2_effnet.pt"
        )
        print(f"[SYSTEM] Loaded weights from MLflow run {SOURCE_WEIGHTS_RUN_ID}")
    except Exception as e:
        print(f"[WARN] Cannot fetch from MLflow: {e}")
        # Try local
        local_path = "best_stage2_effnet.pt"
        if os.path.exists(local_path):
            weights_path = local_path
            print(f"[SYSTEM] Using local weights: {local_path}")

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("[SYSTEM] Pre-trained weights loaded successfully!")
    else:
        print("[WARN] No pre-trained weights found. Training from ImageNet init.")

    # Freeze backbone, only train classifier head
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[SYSTEM] Trainable: {trainable:,} / {total:,} params "
          f"({100*trainable/total:.1f}%)")

    # Loss — only for hardhat (idx 0) and vest (idx 1)
    pos_weight = torch.tensor(
        [(TOTAL_TRAIN - p) / p for p in POS_COUNTS[:2]],
        dtype=torch.float32,
    ).to(device)
    print(f"[SYSTEM] pos_weight (hardhat, vest): {[round(w.item(), 2) for w in pos_weight]}")
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Metrics
    metrics = torchmetrics.MetricCollection({
        'precision': BinaryPrecision(),
        'recall':    BinaryRecall(),
        'f1':        BinaryF1Score(),
    }).to(device)

    best_f1          = 0.0
    early_stop_count = 0
    best_model_path  = "best_stage2_effnet_attention.pt"

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "model":       "tf_efficientnetv2_b0",
            "approach":    "pose_guided_soft_attention",
            "classes":     str(LABEL_COLS[:2]),
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "freeze":      "backbone (classifier only)",
            "sigma":       ATTENTION_SIGMA,
            "floor":       ATTENTION_FLOOR,
            "feat_size":   FEAT_SIZE,
            "patience":    PATIENCE,
        })

        for epoch in range(EPOCHS):
            # --- Train ---
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

            for images, labels, label_mask, head_masks, torso_masks in pbar:
                images      = images.to(device)
                labels      = labels.to(device)
                label_mask  = label_mask.to(device)
                head_masks  = head_masks.to(device)
                torso_masks = torso_masks.to(device)

                optimizer.zero_grad()

                # Masked forward → [B, 2] logits for hardhat & vest
                logits = masked_forward(model, images, head_masks, torso_masks)

                # Labels & mask — only first 2 cols (hardhat, vest)
                target_labels = labels[:, :2]
                target_mask   = label_mask[:, :2]

                loss = (criterion(logits, target_labels) * target_mask).sum() \
                       / target_mask.sum().clamp(min=1.0)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            scheduler.step()

            # --- Val ---
            model.eval()
            val_loss = 0.0
            metrics.reset()

            with torch.no_grad():
                for images, labels, label_mask, head_masks, torso_masks in tqdm(
                        val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    images      = images.to(device)
                    labels      = labels.to(device)
                    label_mask  = label_mask.to(device)
                    head_masks  = head_masks.to(device)
                    torso_masks = torso_masks.to(device)

                    logits       = masked_forward(model, images, head_masks, torso_masks)
                    target_labels = labels[:, :2]
                    target_mask   = label_mask[:, :2]

                    raw_loss = (criterion(logits, target_labels) * target_mask).sum() \
                               / target_mask.sum().clamp(min=1.0)
                    val_loss += raw_loss.item()

                    probs = torch.sigmoid(logits)
                    # Flatten valid predictions for binary metrics
                    valid_preds  = probs[target_mask == 1]
                    valid_labels = target_labels[target_mask == 1]
                    if len(valid_preds) > 0:
                        metrics.update(valid_preds, valid_labels)

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            m         = metrics.compute()
            current_f1 = m['f1'].item()
            current_lr = optimizer.param_groups[0]['lr']

            log_dict = {"train_loss": avg_train, "val_loss": avg_val, "lr": current_lr}
            log_dict.update({f"val_{k}": v.item() for k, v in m.items()})
            mlflow.log_metrics(log_dict, step=epoch)

            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"F1: {current_f1:.4f} | LR: {current_lr:.2e}")

            if current_f1 > best_f1:
                best_f1 = current_f1
                torch.save(model.state_dict(), best_model_path)
                early_stop_count = 0
                print(f"  → Best model saved! F1={best_f1:.4f}")
            else:
                early_stop_count += 1
                if early_stop_count >= PATIENCE:
                    print(f"[SYSTEM] Early stopping at epoch {epoch+1} | Best F1: {best_f1:.4f}")
                    break

        mlflow.log_artifact(best_model_path, artifact_path="weights")
        print(f"[SUCCESS] Done! Best F1: {best_f1:.4f}")


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage2 with Pose-Guided Soft Attention")
    parser.add_argument("--precompute", action="store_true",
                        help="Pre-compute keypoints via YOLO-pose (run once)")
    parser.add_argument("--train", action="store_true",
                        help="Fine-tune classifier head with masked pooling")
    args = parser.parse_args()

    if args.precompute:
        precompute()
    elif args.train:
        train()
    else:
        print("Usage:")
        print("  python train_3_attention.py --precompute   (run once to cache keypoints)")
        print("  python train_3_attention.py --train        (fine-tune classifier head)")
