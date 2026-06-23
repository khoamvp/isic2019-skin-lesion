

# PHẦN 11 — README SPECIFICATION

## 11.1 Mục Tiêu

README phải giúp người khác:

* Hiểu dự án trong 2 phút
* Clone và chạy được
* Hiểu kết quả chính
* Xem Demo nhanh

README phải đủ chất lượng để:

* Portfolio AI Engineer
* GitHub Showcase
* Recruiter Review

---

## 11.2 README Structure

```text
1. Project Overview

2. Dataset

3. Model Architecture

4. Project Structure

5. Installation

6. Training

7. Evaluation

8. Streamlit Demo

9. Results

10. Future Work

11. Author
```

---

## 11.3 Project Overview

Bao gồm:

```markdown
# ISIC 2019 Multimodal Skin Lesion Classification

AI system for skin lesion classification using:

- Dermoscopy Images
- Age
- Sex
- Anatomical Site

Built with:

- PyTorch
- Albumentations
- Streamlit
- Weights & Biases
```

---

## 11.4 Dataset Section

Mô tả:

```text
ISIC 2019
```

Bao gồm:

| Thành phần | Giá trị               |
| ---------- | --------------------- |
| Images     | 25,331                |
| Classes    | 8                     |
| Metadata   | Age, Sex, Anatom Site |
| Source     | Kaggle                |

---

## 11.5 Architecture Section

Chèn sơ đồ:

```text
Image
↓
ResNet50

Metadata
↓
MLP

Concat
↓
Classifier
```

Mô tả ngắn:

```text
Multimodal architecture combining image features and clinical metadata.
```

---

## 11.6 Installation

### Clone Repository

```bash
git clone <repo_url>

cd isic2019-skin-lesion
```

---

### Create Environment

```bash
python -m venv .venv

source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 11.7 Training

### Preprocessing

```bash
python src/preprocessing.py
```

---

### Train

```bash
python src/train.py
```

---

### Evaluate

```bash
python src/evaluate.py
```

---

## 11.8 Streamlit

Run locally:

```bash
streamlit run app.py
```

---

## 11.9 Results Section

Bảng kết quả cuối cùng:

| Model      | Macro F1 | Macro AUC | MEL Recall |
| ---------- | -------- | --------- | ---------- |
| Baseline   | TBD      | TBD       | TBD        |
| Multimodal | TBD      | TBD       | TBD        |

---

## 11.10 Visualizations

Hiển thị:

```text
Confusion Matrix

ROC Curve

Grad-CAM

SHAP
```

---

## 11.11 Future Work

Ví dụ:

```text
Vision Transformer

EfficientNet

Additional Metadata

External Validation Dataset

Docker Deployment

REST API
```

---

## 11.12 Author Section

```markdown
Author: <Tên của bạn>

Role:
AI / Machine Learning Engineer

Focus:
Computer Vision
Medical AI
Multimodal Learning
```

---

# PHẦN 12 — RISK MANAGEMENT

## 12.1 Mục Tiêu

Xác định các rủi ro:

* Kỹ thuật
* Dữ liệu
* Mô hình
* Triển khai

và phương án giảm thiểu.

---

# RISK 1 — Class Imbalance

## Mô tả

Một số lớp rất ít dữ liệu:

```text
DF

VASC

SCC
```

---

## Tác động

```text
Recall thấp

Macro F1 giảm

Model thiên lệch về NV
```

---

## Giảm thiểu

```text
Class Weight

Focal Loss

Macro Metrics
```

---

# RISK 2 — Data Leakage

## Mô tả

Cùng lesion_id xuất hiện ở:

```text
Train

Validation
```

---

## Tác động

```text
Validation ảo

Overestimate hiệu năng
```

---

## Giảm thiểu

```python
StratifiedGroupKFold
```

và:

```python
assert no overlap lesion_id
```

---

# RISK 3 — Metadata Leakage

## Mô tả

Fit StandardScaler trước khi chia Fold.

---

## Tác động

```text
Validation không còn độc lập
```

---

## Giảm thiểu

```text
Split Fold
↓
Fit Scaler trên Train Fold
↓
Transform Validation Fold
```

---

# RISK 4 — Overfitting

## Dấu hiệu

```text
Train Loss giảm

Validation Loss tăng
```

---

## Giảm thiểu

```text
Dropout

Weight Decay

Early Stopping

Data Augmentation
```

---

# RISK 5 — Kaggle Path Change

## Mô tả

Dataset trên Kaggle đổi cấu trúc thư mục.

---

## Tác động

```text
Không đọc được ảnh
```

---

## Giảm thiểu

Kiểm tra:

```python
Path(image_dir).exists()
```

ngay khi khởi tạo notebook.

---

# RISK 6 — GPU Memory Overflow

## Dấu hiệu

```text
CUDA Out Of Memory
```

---

## Giảm thiểu

```text
Batch Size nhỏ hơn

Mixed Precision

Gradient Accumulation
```

---

# RISK 7 — Streamlit Deployment Failure

## Mô tả

Model quá lớn hoặc thiếu dependency.

---

## Giảm thiểu

```text
requirements_deploy.txt riêng

Test local trước deploy

Giảm kích thước checkpoint
```

---

# RISK 8 — Poor MEL Recall

## Mô tả

Recall Melanoma thấp.

---

## Tác động

```text
Bỏ sót ca nguy hiểm
```

---

## Giảm thiểu

```text
Tune Threshold

Focal Loss

Class Weight

Error Analysis
```

---

# RISK 9 — Artifact Mismatch

## Mô tả

Train và Inference dùng metadata khác nhau.

---

## Tác động

```text
Sai số chiều metadata

Crash khi deploy
```

---

## Giảm thiểu

Luôn sử dụng:

```text
feature_cols.pkl
```

---

# RISK 10 — Missing Metadata During Inference

## Mô tả

Người dùng không nhập:

```text
Age

Sex

Anatom Site
```

---

## Giảm thiểu

```text
Age
↓
median_age.pkl

Sex
↓
Unknown

Site
↓
Unknown
```

---

# 12.2 Success Criteria

Dự án được coi là thành công khi:

```text
Macro F1 ≥ 0.75

Macro AUC ≥ 0.92

MEL Recall ≥ 0.85

5-Fold CV hoàn thành

Streamlit Deploy thành công

README hoàn chỉnh

GitHub Public sạch
```

---

# 12.3 Final Deliverables

```text
GitHub Repository

project_plan_final.md

README.md

metadata_processed.csv

median_age.pkl

feature_cols.pkl

best_model.pth

evaluation_report.md

Streamlit App

Public Demo URL
```

---

# END OF DOCUMENT
