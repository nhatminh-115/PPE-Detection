"""
PPE Dataset v2 — with pose-guided attention mask generation.

For each image, runs YOLO pose estimation to get keypoints,
then generates head and torso Gaussian attention masks for
the soft-attention training pipeline.
"""

import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# COCO keypoint indices
_HEAD_KPS    = [0, 1, 2, 3, 4]   # nose, eyes, ears
_TORSO_KPS   = [5, 6, 11, 12]    # shoulders, hips
_KP_CONF_THR = 0.3


def build_gaussian_mask(keypoints, kp_indices, img_w, img_h,
                        feat_h=7, feat_w=7, sigma=0.4, floor=0.2):
    """
    Build a soft Gaussian attention mask on a [feat_h, feat_w] grid.

    For training data, keypoints are in image pixel space (not crop-relative),
    so we normalize by image dimensions.

    Returns: np.ndarray [feat_h, feat_w]
    """
    mask = np.zeros((feat_h, feat_w), dtype=np.float32)
    valid_count = 0

    for idx in kp_indices:
        if idx >= len(keypoints) or keypoints[idx][2] < _KP_CONF_THR:
            continue
        nx = keypoints[idx][0] / max(img_w, 1)
        ny = keypoints[idx][1] / max(img_h, 1)
        fx = nx * feat_w
        fy = ny * feat_h
        s = sigma * max(feat_h, feat_w)
        yy, xx = np.mgrid[0:feat_h, 0:feat_w]
        gaussian = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * s * s))
        mask = np.maximum(mask, gaussian)
        valid_count += 1

    if valid_count == 0:
        # No valid keypoints → uniform mask
        return np.ones((feat_h, feat_w), dtype=np.float32)

    if mask.max() > 1e-4:
        mask = mask / mask.max()
    mask = np.clip(mask * (1.0 - floor) + floor, 0, 1)
    return mask


class PPEAttentionDataset(Dataset):
    """
    Loads full-body images with multi-label targets AND pre-computed pose masks.

    Workflow:
        1. Call `precompute_keypoints()` class method once to run YOLO-pose
           on all images and cache keypoints to a .npz file.
        2. Dataset loads keypoints from cache at runtime and generates masks
           on the fly (fast, purely numpy).
    """
    MIN_CROP_W = 40
    MIN_CROP_H = 80

    def __init__(self, csv_path, img_dir, transform=None, label_cols=None,
                 keypoints_cache=None, feat_size=7, sigma=0.4, floor=0.2):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[FATAL] Cannot find: {csv_path}")

        self.img_dir    = img_dir
        self.transform  = transform
        self.label_cols = label_cols or ['hardhat', 'vest', 'gloves', 'goggles']
        self.feat_size  = feat_size
        self.sigma      = sigma
        self.floor      = floor

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
            except Exception:
                skipped += 1

        self.df = df_raw.loc[valid_indices].reset_index(drop=True)
        print(f"[DATASET] Valid: {len(self.df)} | Skipped: {skipped}")

        # Load pre-computed keypoints {image_name: [17, 3]}
        self.keypoints_map = {}
        if keypoints_cache is not None and os.path.exists(keypoints_cache):
            data = np.load(keypoints_cache, allow_pickle=True)
            self.keypoints_map = data['keypoints'].item()
            print(f"[DATASET] Loaded keypoints cache: {len(self.keypoints_map)} entries")
        else:
            print("[DATASET] WARNING: No keypoints cache. Masks will be uniform.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_name = row['image_name']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"[I/O ERROR] {img_path}: {e}")

        img_w, img_h = image.size

        if self.transform:
            image = self.transform(image)

        labels      = torch.tensor(row[self.label_cols].values.astype('float32'))
        mask        = (labels >= 0).float()
        safe_labels = labels.clamp(min=0)

        # Generate attention masks from cached keypoints
        kps = self.keypoints_map.get(img_name, None)
        if kps is not None:
            head_mask  = build_gaussian_mask(
                kps, _HEAD_KPS, img_w, img_h,
                self.feat_size, self.feat_size, self.sigma, self.floor,
            )
            torso_mask = build_gaussian_mask(
                kps, _TORSO_KPS, img_w, img_h,
                self.feat_size, self.feat_size, self.sigma, self.floor,
            )
        else:
            head_mask  = np.ones((self.feat_size, self.feat_size), dtype=np.float32)
            torso_mask = np.ones((self.feat_size, self.feat_size), dtype=np.float32)

        head_mask_t  = torch.from_numpy(head_mask).unsqueeze(0)   # [1, H, W]
        torso_mask_t = torch.from_numpy(torso_mask).unsqueeze(0)

        return image, safe_labels, mask, head_mask_t, torso_mask_t

    @staticmethod
    def precompute_keypoints(img_dir, csv_path, output_path, pose_model_path,
                             device='cuda:0', batch_size=16):
        """
        Run YOLO-pose on all images and save keypoints to .npz cache.
        Should be called ONCE before training.

        Usage:
            PPEAttentionDataset.precompute_keypoints(
                img_dir='datasets/stage2_ultralytics/train/images',
                csv_path='datasets/stage2_ultralytics/train/labels.csv',
                output_path='datasets/stage2_ultralytics/train/keypoints.npz',
                pose_model_path='yolo26m-pose.pt',
            )
        """
        from ultralytics import YOLO
        import pandas as pd

        df = pd.read_csv(csv_path)
        model = YOLO(pose_model_path)

        keypoints_dict = {}
        img_names = df['image_name'].tolist()

        print(f"[PRE-COMPUTE] Running pose estimation on {len(img_names)} images...")

        for i in range(0, len(img_names), batch_size):
            batch_names = img_names[i:i + batch_size]
            batch_paths = [os.path.join(img_dir, n) for n in batch_names]

            # Filter existing files
            valid_paths = []
            valid_names = []
            for p, n in zip(batch_paths, batch_names):
                if os.path.exists(p):
                    valid_paths.append(p)
                    valid_names.append(n)

            if not valid_paths:
                continue

            results = model.predict(valid_paths, conf=0.3, device=device,
                                    verbose=False, imgsz=640)

            for name, result in zip(valid_names, results):
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    # Take the first (most confident) person's keypoints
                    kps = result.keypoints.data[0].cpu().numpy()  # [17, 3]
                    keypoints_dict[name] = kps
                else:
                    keypoints_dict[name] = np.zeros((17, 3), dtype=np.float32)

            if (i // batch_size) % 10 == 0:
                print(f"  [{i}/{len(img_names)}] processed")

        np.savez_compressed(output_path, keypoints=keypoints_dict)
        print(f"[PRE-COMPUTE] Saved {len(keypoints_dict)} keypoints → {output_path}")


def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
