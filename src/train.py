"""
train.py — M2 Baseline Model Training
--------------------------------------
- Đọc metadata_processed.csv + feature_cols.pkl
- Fit StandardScaler trên Train Fold
- Train BaselineModel (ResNet50, image-only) 1 fold
- Focal Loss + Class Weights
- Early stopping theo val Macro-F1
- W&B logging + checkpoint best_baseline.pth
"""

import os
import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, recall_score

import wandb
from dotenv import load_dotenv

from src.model import BaselineModel
from src.dataset import ISICDataset, TRAIN_TRANSFORM, VAL_TRANSFORM, make_dataloader

# =====================================================
# CONFIG
# =====================================================

CONFIG = {
    "model"        : "baseline",
    "backbone"     : "resnet50",
    "lr"           : 1e-4,
    "batch_size"   : 32,
    "epochs"       : 30,
    "patience"     : 7,
    "val_fold"     : 1,        # fold 0 = test (khóa), fold 1 = val cho M2
    "n_folds"      : 5,
    "seed"         : 42,
    "loss"         : "focal",
    "img_size"     : 224,
    "grad_clip"    : 1.0,
    "num_workers"  : 0,        # 0 = Windows; 2 = Kaggle/Linux
    "num_classes"  : 8,
}

LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Tự động phát hiện môi trường: Nếu có thư mục /kaggle -> LOCAL = False
LOCAL = not os.path.exists("/kaggle")
IMAGE_DIR = Path("data/raw/train") if LOCAL else Path("/kaggle/input/isic-2019-dataset-full/isic-2019-dataset-full/train")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# =====================================================
# SEED
# =====================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================
# FOCAL LOSS
# =====================================================

class FocalLoss(nn.Module):
    """
    Focal Loss cho multi-class classification.
    Phạt nặng khi model tự tin sai ở lớp khó/hiếm.
    alpha: class weights (tensor shape [num_classes])
    gamma: focusing parameter (default 2.0)
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha          # class weights
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)                          # (batch,)
        pt = torch.exp(-ce_loss)                                    # prob của class đúng
        focal_loss = (1 - pt) ** self.gamma * ce_loss              # focal weighting
        return focal_loss.mean() if self.reduction == "mean" else focal_loss

# =====================================================
# CLASS WEIGHTS
# =====================================================

def compute_class_weights(labels: pd.Series, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Tính class weights nghịch đảo tần suất.
    Lớp hiếm → weight cao → model bị phạt nặng hơn khi sai.
    """
    counts = labels.value_counts().sort_index()
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * num_classes     # normalize về ~1.0 trung bình
    return torch.tensor(weights, dtype=torch.float32).to(device)

# =====================================================
# TRAIN 1 EPOCH
# =====================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, _meta, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)                    # BaselineModel: chỉ nhận ảnh
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1

# =====================================================
# VALIDATE
# =====================================================

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, _meta, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / len(loader.dataset)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mel_recall = recall_score(all_labels, all_preds, labels=[0], average="macro", zero_division=0)

    # Macro-AUC: chỉ tính nếu đủ 8 lớp trong val
    try:
        macro_auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro"
        )
    except ValueError:
        macro_auc = 0.0

    accuracy = (all_preds == all_labels).mean()

    return {
        "loss"        : avg_loss,
        "macro_f1"    : macro_f1,
        "macro_auc"   : macro_auc,
        "mel_recall"  : mel_recall,
        "accuracy"    : accuracy,
    }

# =====================================================
# SCALE METADATA (fit trên train, transform cả 2)
# =====================================================

def fit_scaler(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list):
    """
    Fit StandardScaler CHỈ trên train_df.
    Transform cả train_df và val_df.
    Lưu scaler.pkl để dùng khi inference.
    """
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df   = val_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("[SCALER] Fit xong, lưu models/scaler.pkl")

    return train_df, val_df

# =====================================================
# MAIN TRAINING LOOP
# =====================================================

def main():
    set_seed(CONFIG["seed"])
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # --- Load data ---
    df = pd.read_csv("data/processed/metadata_processed.csv")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    print(f"[DATA] rows={len(df):,} | meta_dim={len(feature_cols)}")

    # --- Split: fold 0 = test (khóa), fold VAL_FOLD = val, còn lại = train ---
    val_fold = CONFIG["val_fold"]
    train_df = df[(df["fold"] != 0) & (df["fold"] != val_fold)].reset_index(drop=True)
    val_df   = df[df["fold"] == val_fold].reset_index(drop=True)
    print(f"[SPLIT] train={len(train_df):,} | val={len(val_df):,}")

    # --- Fit StandardScaler trên train ---
    train_df, val_df = fit_scaler(train_df, val_df, feature_cols)

    # --- Class weights ---
    class_weights = compute_class_weights(train_df["label"], CONFIG["num_classes"], device)
    print(f"[WEIGHTS] {class_weights.cpu().numpy().round(3)}")

    # --- Datasets & Dataloaders ---
    train_dataset = ISICDataset(train_df, IMAGE_DIR, feature_cols, transform=TRAIN_TRANSFORM)
    val_dataset   = ISICDataset(val_df,   IMAGE_DIR, feature_cols, transform=VAL_TRANSFORM)

    train_loader = make_dataloader(train_dataset, train_df, is_train=True,
                                   batch_size=CONFIG["batch_size"],
                                   num_workers=CONFIG["num_workers"])
    val_loader   = make_dataloader(val_dataset, val_df, is_train=False,
                                   batch_size=CONFIG["batch_size"],
                                   num_workers=CONFIG["num_workers"])

    # --- Model ---
    model = BaselineModel(num_classes=CONFIG["num_classes"]).to(device)

    # --- Loss & Optimizer ---
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # --- W&B init ---
    wandb.init(
        project="isic2019-skin-lesion",
        name=f"baseline_resnet50_fold{val_fold}",
        config=CONFIG,
        mode=os.getenv("WANDB_MODE", "online"),   # set WANDB_MODE=offline nếu mất mạng
    )

    # --- Training loop ---
    best_f1       = 0.0
    patience_cnt  = 0
    best_ckpt     = MODELS_DIR / "best_baseline.pth"

    print(f"\n[TRAIN] Bắt đầu training — {CONFIG['epochs']} epochs\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, CONFIG["grad_clip"]
        )
        val_metrics = validate(model, val_loader, criterion, device, CONFIG["num_classes"])
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        # Log W&B
        wandb.log({
            "epoch"               : epoch,
            "train/loss"          : train_loss,
            "train/macro_f1"      : train_f1,
            "train/learning_rate" : current_lr,
            "val/loss"            : val_metrics["loss"],
            "val/accuracy"        : val_metrics["accuracy"],
            "val/macro_f1"        : val_metrics["macro_f1"],
            "val/macro_auc"       : val_metrics["macro_auc"],
            "val/mel_recall"      : val_metrics["mel_recall"],
        })

        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
            f"val_auc={val_metrics['macro_auc']:.4f} mel_recall={val_metrics['mel_recall']:.4f}"
        )

        # --- Checkpoint khi val F1 cải thiện ---
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            patience_cnt = 0
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_f1"    : best_f1,
                "config"         : CONFIG,
            }, best_ckpt)
            print(f"  → [SAVED] best_baseline.pth | val_f1={best_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f"\n[EARLY STOP] Không cải thiện sau {CONFIG['patience']} epoch. Dừng lại.")
                break

    # --- Kết quả cuối ---
    print(f"\n[DONE] Best val Macro-F1 = {best_f1:.4f}")
    if best_f1 < 0.60:
        print("[WARNING] Macro-F1 < 0.60 — Kiểm tra lại data pipeline trước khi sang M3!")
    else:
        print("[OK] Đạt ngưỡng M2. Sẵn sàng sang M3.")

    wandb.summary["best_val_macro_f1"] = best_f1
    wandb.finish()


if __name__ == "__main__":
    main()