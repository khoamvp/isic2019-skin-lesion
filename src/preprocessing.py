import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# Tự động phát hiện môi trường: Nếu có thư mục /kaggle -> LOCAL = False
LOCAL = not os.path.exists("/kaggle")

LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = "data/raw" if LOCAL else "/kaggle/input/datasets/alifshahariar/isic-2019-dataset-full/isic-2019-dataset-full"

# Hàm gộp dữ liệu metadata và nhãn (ground truth)
def load_and_merge(meta_path: str, gt_path: str) -> pd.DataFrame:
    df_meta = pd.read_csv(meta_path)
    df_gt = pd.read_csv(gt_path)
    # Gộp theo cột 'image'
    df = df_meta.merge(df_gt, on="image", how="inner")
    
    # Lấy vị trí index có xác suất cao nhất làm label chính
    df["label"] = df[LABEL_COLS].values.argmax(axis=1)
    assert df["image"].nunique() == len(df), "Có image bị trùng sau khi merge."
    print(f"[LOAD] rows={len(df):,} | classes={df['label'].nunique()}")
    
    return df

# Hàm xử lý các giá trị bị thiếu (NaN) - phần KHÔNG phụ thuộc train/test split
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
    df["sex"] = df["sex"].fillna("unknown")
    df["anatom_site_general"] = df["anatom_site_general"].fillna("unknown")
    
    # SỬA LỖI: Điền giá trị thiếu bằng chính tên ảnh để không làm lệch thuật toán chia nhóm nhóm
    df["lesion_id"] = df["lesion_id"].fillna(df["image"]).astype(str)

    return df

# Hàm fillna age_approx - PHẢI gọi SAU khi đã có train_df/test_df
def fill_age(trainval_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    trainval_df = trainval_df.copy()
    test_df = test_df.copy()

    median_age = trainval_df["age_approx"].median()

    trainval_df["age_approx"] = trainval_df["age_approx"].fillna(median_age)
    test_df["age_approx"] = test_df["age_approx"].fillna(median_age)

    assert trainval_df["age_approx"].isnull().sum() == 0
    assert test_df["age_approx"].isnull().sum() == 0

    return trainval_df, test_df, median_age

# Hàm biến đổi các cột chữ thành số (One-Hot Encoding)
def encode_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    df = pd.get_dummies(df, columns=["sex", "anatom_site_general"], dtype=float)
    
    # Lưu lại danh sách các cột feature mới được tạo ra
    feature_cols = ["age_approx"] + [c for c in df.columns if c.startswith("sex_") or c.startswith("anatom_site_")]
    print(f"[ENCODE] feature_dim={len(feature_cols)}")
    
    return df, feature_cols

# Hàm chia tập dữ liệu thành các fold để huấn luyện (Cross Validation)
def create_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    
    # Group: Tránh rò rỉ dữ liệu (data leakage) giữa các ảnh của cùng 1 tổn thương da (lesion_id)
    # Stratify: Giữ tỷ lệ các nhãn (label) cân bằng giữa các fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(sgkf.split(df, y=df["label"], groups=df["lesion_id"])):
        df.loc[val_idx, "fold"] = fold

    assert (df["fold"] >= 0).all()
    print(f"[FOLD] {n_splits} folds created")
    
    return df

# Hàm lưu các giá trị tham chiếu để tái sử dụng khi test/inference
def save_artifacts(median_age: float, feature_cols: list):
    joblib.dump(median_age, MODELS_DIR / "median_age.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print("[SAVE] median_age.pkl & feature_cols.pkl")

if __name__ == "__main__":
    print(f"--- Đang khởi động Data Pipeline với môi trường: {'LOCAL' if LOCAL else 'KAGGLE'} ---")
    
    TRAIN_META = f"{DATA_DIR}/ISIC_2019_Training_Metadata.csv"
    TRAIN_GT = f"{DATA_DIR}/ISIC_2019_Training_GroundTruth.csv"

    # Pipeline xử lý dữ liệu chuẩn bị cho Model
    df = load_and_merge(TRAIN_META, TRAIN_GT)
    df = fill_missing(df)                          # đã sửa logic điền lesion_id
    df, feature_cols = encode_metadata(df)         # one-hot sex/anatom; age vẫn raw
    df = create_folds(df, n_splits=5, seed=42)     # chia fold an toàn, không còn lỗi mixed-type hay imbalanced

    # Tách trainval/test để fillna age đúng thứ tự (chống leakage)
    trainval_df = df[df["fold"] != 0].reset_index(drop=True)
    test_df     = df[df["fold"] == 0].reset_index(drop=True)
    trainval_df, test_df, median_age = fill_age(trainval_df, test_df)
    df = pd.concat([trainval_df, test_df], ignore_index=True)

    # Kiểm tra rò rỉ bệnh nhân giữa Fold 0 (làm tập Test mẫu) và các Fold còn lại (tập Train)
    train_lesions = set(df[df["fold"] != 0]["lesion_id"])
    test_lesions = set(df[df["fold"] == 0]["lesion_id"])
    assert len(train_lesions.intersection(test_lesions)) == 0, "LỖI: Data Leakage! Bệnh nhân bị rò rỉ giữa train và test."
    
    # Bỏ cột lesion_id sau khi split xong (Không cần đưa vào DataLoader)
    df = df.drop(columns=["lesion_id"])
    
    # Lưu lại DataFrame đã xử lý dưới dạng CSV
    output_csv = PROCESSED_DIR / "metadata_processed.csv"
    df.to_csv(output_csv, index=False)
    print(f"[SAVE] {output_csv}")

    # Lưu file .pkl
    save_artifacts(median_age, feature_cols)
    print(f"[THÀNH CÔNG] Toàn bộ dữ liệu sạch đã được xuất bản tại: {output_csv}")
    print(f"[DONE] rows={len(df):,} | meta_dim={len(feature_cols)}")