import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PPEMultiLabelDataset(Dataset):
    MIN_CROP_W = 40
    MIN_CROP_H = 80

    def __init__(self, csv_path, img_dir, transform=None, label_cols=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[FATAL] Khong tim thay: {csv_path}")
        self.img_dir    = img_dir
        self.transform  = transform
        self.label_cols = label_cols or ['hardhat', 'vest', 'gloves', 'goggles']  # bo mask

        df_raw = pd.read_csv(csv_path)
        print(f"[DATASET] Raw: {len(df_raw)} | Classes: {self.label_cols}")

        valid_indices, skipped = [], 0
        for i, row in df_raw.iterrows():
            try:
                w, h = Image.open(os.path.join(img_dir, row['image_name'])).size
                if w >= self.MIN_CROP_W and h >= self.MIN_CROP_H:
                    valid_indices.append(i)
                else:
                    skipped += 1
            except:
                skipped += 1

        self.df = df_raw.loc[valid_indices].reset_index(drop=True)
        print(f"[DATASET] Valid: {len(self.df)} | Skipped: {skipped}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"[I/O ERROR] {img_path}: {e}")
        if self.transform:
            image = self.transform(image)
        labels      = torch.tensor(row[self.label_cols].values.astype('float32'))
        mask        = (labels >= 0).float()
        safe_labels = labels.clamp(min=0)
        return image, safe_labels, mask

def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])