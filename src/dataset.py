import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================================================
# IMAGE TRANSFORMS
# =====================================================

TRAIN_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Augmentation mạnh hơn cho lớp thiểu số (AK, DF, VASC, SCC)
TRAIN_TRANSFORM_HEAVY = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.Affine(shear=(-15, 15), p=0.3),
    A.GridDistortion(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

INFERENCE_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

VAL_TRANSFORM = INFERENCE_TRANSFORM

# =====================================================
# DATASET
# =====================================================

# Các lớp thiểu số cần augment mạnh hơn
MINORITY_CLASSES = {2, 4, 6, 7}  # AK=2, DF=4, VASC=6, SCC=7 (theo thứ tự label)

class ISICDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, feature_cols: list, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.feature_cols = feature_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Đọc ảnh — check None TRƯỚC khi cvtColor
        image_path = self.image_dir / f"{row['image']}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Cannot load image: {image_path}. Using blank image.")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Chọn transform theo lớp (nếu đang train)
        label_int = int(row["label"])
        if self.transform is not None:
            # Lớp thiểu số dùng heavy transform khi train
            if label_int in MINORITY_CLASSES and self.transform == TRAIN_TRANSFORM:
                image = TRAIN_TRANSFORM_HEAVY(image=image)["image"]
            else:
                image = self.transform(image=image)["image"]

        # 3. Metadata
        metadata = torch.tensor(
            row[self.feature_cols].values.astype(np.float32),
            dtype=torch.float32
        )

        # 4. Label
        label = torch.tensor(label_int, dtype=torch.long)

        return image, metadata, label

# =====================================================
# DATALOADER
# =====================================================

def make_dataloader(dataset, df, is_train=True, use_sampler=False, batch_size=32, num_workers=2):
    # is_train=True, use_sampler=False (mặc định): shuffle=True, Focal Loss + Class Weights ở train.py
    # use_sampler=True : CHỈ bật khi sau M2 model vẫn predict toàn NV (fallback, theo 5.6)
    if is_train and use_sampler:
        class_counts = df['label'].value_counts().to_dict()
        weights = [1.0 / class_counts[int(label)] for label in df['label']]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True
        )

# =====================================================
# SANITY CHECK
# =====================================================

def sanity_check(dataset, df, n_samples=50):
    print("\n===== SANITY CHECK =====")

    # --- Check 1 sample trước ---
    image, meta, label = dataset[0]
    assert image.shape == (3, 224, 224), f"Lỗi chiều ảnh: {image.shape}"
    assert meta.ndim == 1,               "Metadata phải là 1D"
    assert image.dtype == torch.float32, "Ảnh phải float32"
    assert meta.dtype  == torch.float32, "Meta phải float32"
    assert label.dtype == torch.int64,   "Label phải int64"
    print("[OK] Single sample passed")

    # --- Tạo loader 50 ảnh để check batch ---
    from torch.utils.data import DataLoader, Subset
    subset = Subset(dataset, indices=list(range(min(n_samples, len(dataset)))))
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    imgs, metas, labels = next(iter(loader))

    # [ ] img_tensor.shape == (32, 3, 224, 224)
    assert imgs.shape == (32, 3, 224, 224), f"Lỗi batch ảnh: {imgs.shape}"
    print(f"[OK] img_tensor.shape   : {imgs.shape}")

    # [ ] meta_tensor.shape == (32, n_features)
    assert metas.ndim == 2, f"Meta phải 2D: {metas.shape}"
    print(f"[OK] meta_tensor.shape  : {metas.shape}")

    # [ ] label.shape == (32,)
    assert labels.shape == (32,), f"Lỗi label shape: {labels.shape}"
    print(f"[OK] label.shape        : {labels.shape}")

    # [ ] dtype
    assert imgs.dtype   == torch.float32, "img phải float32"
    assert metas.dtype  == torch.float32, "meta phải float32"
    assert labels.dtype == torch.int64,   "label phải int64"
    print(f"[OK] dtypes             : img={imgs.dtype} | meta={metas.dtype} | label={labels.dtype}")

    # [ ] img pixel range ≈ [-2.5, 2.5] sau normalize
    pmin, pmax = imgs.min().item(), imgs.max().item()
    assert -3.0 < pmin < 0,    f"Pixel min bất thường: {pmin:.3f}"
    assert  0   < pmax < 3.0,  f"Pixel max bất thường: {pmax:.3f}"
    print(f"[OK] pixel range        : [{pmin:.3f}, {pmax:.3f}]")
    if len(df['label'].unique()) == 8:
        # [ ] 8 lớp đều xuất hiện trong 1 epoch (dùng WeightedRandomSampler)
        full_loader = make_dataloader(dataset, df, is_train=True)
        all_labels = []
        for i, (_, _, lbl) in enumerate(full_loader):
            all_labels.extend(lbl.tolist())
            if i >= 200:
                break
        unique_classes = set(all_labels)
        assert len(unique_classes) == 8, f"Thiếu lớp: chỉ thấy {unique_classes}"
        (f"[OK] 8 lớp xuất hiện   : {sorted(unique_classes)}")
    else:
        print(f"[SKIP] Chỉ có {len(df['label'].unique())} lớp trong sample — bỏ qua check 8 lớp")

    print("\n===== ALL CHECKS PASSED =====")

# =====================================================
# TEST SCRIPT
# =====================================================

if __name__ == "__main__":
    import joblib

    IMAGE_DIR = Path("data/raw/train")
    df = pd.read_csv("data/processed/metadata_processed.csv")
    feature_cols = joblib.load("models/feature_cols.pkl")

    # Lọc chỉ giữ dòng có ảnh thực tế trên disk
    existing = [f.stem for f in IMAGE_DIR.glob("*.jpg")]
    df_sample = df[df['image'].isin(existing)].reset_index(drop=True)
    print(f"Ảnh có thật: {len(df_sample)} / {len(df)}")

    train_df = df_sample[df_sample["fold"] != 0].reset_index(drop=True)

    dataset = ISICDataset(
        df=train_df,
        image_dir=IMAGE_DIR,
        feature_cols=feature_cols,
        transform=TRAIN_TRANSFORM
    )

    sanity_check(dataset, train_df)