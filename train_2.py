import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
import mlflow, timm, torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryCalibrationError
from tqdm import tqdm
from dotenv import load_dotenv
from ppe_dataset import PPEMultiLabelDataset, get_val_transforms

load_dotenv()

# ==============================================================================
# CONFIG
# ==============================================================================
DAGSHUB_URI     = "https://dagshub.com/nhatminh-115/PPE-Detection.mlflow"
EXPERIMENT_NAME = "PPE_Stage2_Classification_"
RUN_NAME        = "EfficientNetV2_B0_4Class_1"

CSV_TRAIN = r"datasets\stage2_ultralytics\train\labels.csv"
IMG_TRAIN = r"datasets\stage2_ultralytics\train\images"
CSV_VAL   = r"datasets\stage2_ultralytics\val\labels.csv"
IMG_VAL   = r"datasets\stage2_ultralytics\val\images"

# Bo mask (chi 1 sample) → 4 class
LABEL_COLS  = ['hardhat', 'vest', 'gloves', 'goggles']
NUM_CLASSES = 4
IMG_SIZE    = 224
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 1e-4

TOTAL_TRAIN = 1755
POS_COUNTS  = [1202, 1209, 628, 449]  # hardhat, vest, gloves, goggles

PATIENCE    = 10   # overnight → generous

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

def mixup_data(x, y, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

# ==============================================================================
# TRAIN
# ==============================================================================
def train_stage2():
    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Device: {device}")

    train_ds = PPEMultiLabelDataset(CSV_TRAIN, IMG_TRAIN, get_train_transforms(IMG_SIZE), LABEL_COLS)
    val_ds   = PPEMultiLabelDataset(CSV_VAL,   IMG_VAL,   get_val_transforms(IMG_SIZE),   LABEL_COLS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # B0 + freeze 50% backbone
    print("[SYSTEM] Load EfficientNetV2-B0, freeze 50% backbone...")
    model  = timm.create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=NUM_CLASSES).to(device)
    layers = list(model.children())
    for layer in layers[:int(len(layers) * 0.5)]:
        for param in layer.parameters():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SYSTEM] Trainable params: {trainable:,}")

    # Loss
    pos_weight = torch.tensor([(TOTAL_TRAIN - p) / p for p in POS_COUNTS], dtype=torch.float32).to(device)
    print(f"[SYSTEM] pos_weight: {[round(w.item(),2) for w in pos_weight]}")
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    # Optimizer + CosineAnnealing (tốt hơn ReduceLROnPlateau cho long run)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Metrics
    metrics = torchmetrics.MetricCollection({
        'precision': BinaryPrecision(),
        'recall':    BinaryRecall(),
        'f1':        BinaryF1Score(),
        'ece':       BinaryCalibrationError(n_bins=10),
    }).to(device)

    best_f1           = 0.0
    early_stop_count  = 0
    best_model_path   = "best_stage2_effnet.pt"

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "model": "tf_efficientnetv2_b0",
            "classes": str(LABEL_COLS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "freeze": "50%",
            "augmentation": "strong+mixup",
            "scheduler": "CosineAnnealing",
            "patience": PATIENCE,
        })

        for epoch in range(EPOCHS):
            # --- Train ---
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for images, labels, mask in pbar:
                images, labels, mask = images.to(device), labels.to(device), mask.to(device)
                images, la, lb, lam  = mixup_data(images, labels)

                optimizer.zero_grad()
                outputs = model(images)
                loss_a  = (criterion(outputs, la) * mask).sum() / mask.sum().clamp(min=1.0)
                loss_b  = (criterion(outputs, lb) * mask).sum() / mask.sum().clamp(min=1.0)
                loss    = lam * loss_a + (1 - lam) * loss_b
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
                for images, labels, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    images, labels, mask = images.to(device), labels.to(device), mask.to(device)
                    outputs  = model(images)
                    raw_loss = (criterion(outputs, labels) * mask).sum() / mask.sum().clamp(min=1.0)
                    val_loss += raw_loss.item()
                    probs = torch.sigmoid(outputs)
                    metrics.update(probs[mask == 1], labels[mask == 1])

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            m         = metrics.compute()
            current_f1 = m['f1'].item()
            current_lr = optimizer.param_groups[0]['lr']

            log_dict = {"train_loss": avg_train, "val_loss": avg_val, "lr": current_lr}
            log_dict.update({f"val_{k}": v.item() for k, v in m.items()})
            mlflow.log_metrics(log_dict, step=epoch)

            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"F1: {current_f1:.4f} | ECE: {m['ece']:.4f} | LR: {current_lr:.2e}")

            # Save best by F1 (ổn định hơn val_loss)
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

if __name__ == "__main__":
    train_stage2()