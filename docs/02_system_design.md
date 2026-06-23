
# PHẦN 3 — TECHNICAL DESIGN

## 3.1 Kiến trúc tổng thể

```text
Image
↓
CNN Branch
↓
Feature Vector

Metadata
↓
MLP Branch
↓
Feature Vector

Concat
↓
Classifier
↓
8 Classes
```

---

## 3.2 CNN Branch

Backbone:

```text
ResNet50 Pretrained
```

Lý do:

* Mạnh
* Ổn định
* Dễ Grad-CAM
* Phù hợp Kaggle GPU

Output:

```text
2048-dimensional vector
```

---

## 3.3 Metadata Branch

Input:

* Age
* Sex
* Anatom Site

Kiến trúc:

```text
Linear
↓
BatchNorm
↓
ReLU
↓
Dropout
↓
Linear
```

Output:

```text
16-dimensional vector
```

---

## 3.4 Fusion Layer

```text
2048
+
16
=
2064
```

Sau đó:

```text
Linear(2064 → 8)
```

---

## 3.5 Explainability

### Grad-CAM

Mục tiêu:

* Xác định vùng ảnh ảnh hưởng tới dự đoán.

---

### SHAP

Mục tiêu:

* Đánh giá đóng góp của metadata.

---

## 3.6 Tech Stack

| Thành phần       | Công nghệ        |
| ---------------- | ---------------- |
| Deep Learning    | PyTorch          |
| Augmentation     | Albumentations   |
| Tracking         | Weights & Biases |
| Dataset          | Kaggle Dataset   |
| Model Versioning | DVC              |
| Deployment       | Streamlit        |
| Source Control   | GitHub           |

---

## 3.7 Model Comparison

Thực hiện 2 mô hình:

### Baseline

```text
ResNet50
(Image Only)
```

---

### Multimodal

```text
ResNet50
+
Metadata MLP
```

---

So sánh bằng:

* Macro F1
* Macro AUC
* MEL Recall
* McNemar Test

---

# PHẦN 4 — ENVIRONMENT SETUP

## 4.1 Folder Structure

```text
isic2019-skin-lesion/

├── data/
│   └── processed/
│       └── metadata_processed.csv

├── docs/
│   ├── project_plan_final.md
│   └── experiment_log.md

├── models/
│   ├── median_age.pkl
│   ├── feature_cols.pkl
│   └── best_model.pth

├── notebooks/

├── reports/
│   └── figures/

├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py

├── app.py

├── requirements.txt

├── requirements_deploy.txt

├── README.md

└── .gitignore
```

---

## 4.2 Dataset Strategy

Dataset không được lưu trong GitHub.

Nguồn dữ liệu:

```text
Kaggle Dataset
```

Ảnh được đọc trực tiếp từ:

```text
/input/isic-2019-dataset-full/
```

---

## 4.3 DVC Strategy

DVC chỉ dùng để quản lý:

```text
best_model.pth
```

Không dùng DVC cho:

* Dataset
* CSV
* Ảnh

---

## 4.4 Git Ignore

```text
__pycache__/
.ipynb_checkpoints/

*.pth
*.pt

.env

wandb/

data/

models/
```

---

## 4.5 Setup Checklist

```text
[X] Tạo Repository GitHub

[X] Clone Repository

[X] Tạo môi trường Python

[X] Cài requirements.txt

[X] Tạo Project W&B

[X] Kết nối Kaggle Dataset

[X] Tạo notebook Kaggle

[X] Tạo cấu trúc thư mục

[X] Commit lần đầu

[X] Tạo project_plan_final.md
```