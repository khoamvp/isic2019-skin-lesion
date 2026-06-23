

# PHẦN 6 — EXPERIMENT TRACKING

## 6.1 Công Cụ

Sử dụng:

```text
Weights & Biases
(W&B)
```

---

## 6.2 Run Naming Convention

Ví dụ:

```text
baseline_fold1

baseline_fold2

multimodal_fold1

multimodal_fold2

multimodal_threshold035
```

---

## 6.3 Logging Theo Epoch

Log:

```python
train_loss

val_loss

accuracy

macro_f1

macro_auc

mel_recall

mel_precision

learning_rate
```

---

## 6.4 Checkpoint Strategy

Lưu khi:

```text
Best Validation Macro-F1
```

File:

```text
best_model.pth
```

---

## 6.5 Config Tracking

Lưu toàn bộ:

```python
batch_size

lr

epochs

seed

loss

optimizer

scheduler

backbone
```

---
## 6.6 Experiment Comparison

| Run        | Macro F1 | Macro AUC | MEL Recall |
| ---------- | -------- | --------- | ---------- |
| Baseline   |          |           |            |
| Multimodal |          |           |            |

---

## 6.7 Điều Kiện Thành Công

```text
Macro F1 ≥ 0.75

Macro AUC ≥ 0.92

MEL Recall ≥ 0.85

McNemar p < 0.05
```

---

## 6.8 Experiment Log

File:

```text
docs/experiment_log.md
```

Mẫu:

```markdown
Run:

Date:

Config:

Result:

Observation:

Next Step:
```
# PHẦN 7 — MILESTONE & DONE CRITERIA

## 7.1 Tổng Quan

```text
M0 → Environment Setup

M1 → Data Pipeline

M2 → Baseline Model

M3 → Multimodal Model

M4 → Evaluation

M5 → Streamlit App

M6 → Deploy & Portfolio
```

---

# M0 — Environment Setup

**Thời gian:** 1 ngày

## Mục tiêu

Chuẩn bị toàn bộ môi trường phát triển.

### Done Criteria

```text
[x] GitHub Repository đã tạo

[x] Project Structure hoàn chỉnh

[x] requirements.txt hoạt động

[x] Kaggle Notebook tạo thành công

[x] Kaggle Dataset kết nối thành công

[x] W&B Project hoạt động

[x] .gitignore hoạt động đúng

[x] project_plan_final.md hoàn chỉnh
```

---

# M1 — Data Pipeline

**Thời gian:** 2–3 ngày

## Mục tiêu

Xây dựng pipeline xử lý dữ liệu hoàn chỉnh.

### Done Criteria

```text
[x] Metadata merge thành công

[x] Label column tạo đúng

[x] Missing Values xử lý hoàn tất

[x] median_age.pkl được lưu

[x] feature_cols.pkl được lưu

[x] metadata_processed.csv được tạo

[x] StratifiedGroupKFold hoạt động

[x] Không có lesion_id trùng

[x] Dataset Class hoạt động

[x] Sanity Check pass
```

---

# M2 — Baseline Model

**Thời gian:** 2–3 ngày

## Mục tiêu

Huấn luyện ResNet50 chỉ dùng ảnh.

### Kiến trúc

```text
Image
↓
ResNet50
↓
Classifier
↓
8 Classes
```

### Done Criteria

```text
[ ] BaselineModel hoàn chỉnh

[ ] Train 1 Fold thành công

[ ] Validation hoạt động

[ ] W&B log đầy đủ

[ ] best_model.pth được lưu

[ ] Confusion Matrix được tạo

[ ] Macro-F1 > 0.60
```

---

# M3 — Multimodal Model

**Thời gian:** 3–5 ngày

## Mục tiêu

Kết hợp ảnh và metadata.

### Kiến trúc

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

### Done Criteria

```text
[ ] MultimodalNet hoàn chỉnh

[ ] Metadata branch hoạt động

[ ] Fusion layer hoạt động

[ ] Focal Loss hoạt động

[ ] Class Weight hoạt động

[ ] Train 5 Fold thành công

[ ] Delta Macro-F1 ≥ 3%

[ ] MEL Recall ≥ 85%
```

---

# M4 — Evaluation

**Thời gian:** 2–3 ngày

## Mục tiêu

Đánh giá mô hình trên tập Test.

### Done Criteria

```text
[ ] Macro-AUC ≥ 0.92

[ ] Macro-F1 ≥ 0.75

[ ] MEL Recall ≥ 0.85

[ ] McNemar Test hoàn thành

[ ] 95% CI hoàn thành

[ ] Grad-CAM hoạt động

[ ] SHAP hoạt động

[ ] Evaluation Report hoàn chỉnh
```

---

# M5 — Streamlit App

**Thời gian:** 3–4 ngày

## Mục tiêu

Xây dựng giao diện suy luận.

### Done Criteria

```text
[ ] Upload Image

[ ] Nhập Metadata

[ ] Inference thành công

[ ] Probability Table

[ ] Grad-CAM

[ ] Risk Label

[ ] Clinical Report

[ ] Missing Metadata không crash

[ ] Inference < 3 giây
```

---

# M6 — Deploy & Portfolio

**Thời gian:** 1–2 ngày

## Done Criteria

```text
[ ] Deploy Streamlit Cloud

[ ] README hoàn chỉnh

[ ] GIF Demo

[ ] GitHub sạch

[ ] Không commit dataset

[ ] Không commit checkpoint nhầm

[ ] Public Demo hoạt động
```

---

## 7.2 Tổng Thời Gian

```text
M0 → 1 ngày

M1 → 2–3 ngày

M2 → 2–3 ngày

M3 → 3–5 ngày

M4 → 2–3 ngày

M5 → 3–4 ngày

M6 → 1–2 ngày
```

---

**Tổng cộng:**

```text
14–21 ngày
(part-time 3–4h/ngày)
```

---

## 7.3 Quy Tắc Chuyển Milestone

Không chuyển milestone nếu chưa đạt Done Criteria.

Ví dụ:

```text
M2 chưa đạt Macro-F1 > 0.60

→ Không được chuyển sang M3
```

---

## 7.4 Quy Tắc Test Set

Tập Test:

```text
Chỉ được mở ở M4
```

Không sử dụng để:

* Tune Hyperparameter
* Chọn Model
* Chọn Threshold

---

# PHẦN 8 — EVALUATION PLAN

## 8.1 Mục Tiêu Đánh Giá

Dự án không tối ưu Accuracy đơn thuần.

Ưu tiên:

```text
MEL Recall
↓
Macro AUC
↓
Macro F1
↓
Accuracy
```

---

## 8.2 Metrics Tổng Thể

| Metric    | Mục tiêu |
| --------- | -------- |
| Macro AUC | ≥ 0.92   |
| Macro F1  | ≥ 0.75   |
| Accuracy  | ≥ 0.85   |

---

## 8.3 Metrics Lâm Sàng

### MEL Recall

KPI quan trọng nhất.

```text
Recall ≥ 0.85
```

Lý do:

```text
False Negative
=
Bỏ sót ung thư
```

---

### MEL Precision

Mục tiêu:

```text
Càng cao càng tốt
```

---

### MEL F1

Mục tiêu:

```text
≥ 0.75
```

---

### NV Specificity

Mục tiêu:

```text
≥ 0.80
```

---

## 8.4 Metrics Theo Từng Lớp

Tính:

```python
F1 per Class
```

Cho:

```text
MEL

NV

BCC

AK

BKL

DF

VASC

SCC
```

---

## 8.5 ROC-AUC

### One-vs-Rest

Tính ROC cho từng lớp.

Output:

```text
roc_curve_per_class.png
```

---

## 8.6 Confusion Matrix

Output:

```text
confusion_matrix.png
```

Phân tích:

### Top-3 lỗi nhiều nhất

Ví dụ:

```text
MEL → NV

BCC → BKL

AK → SCC
```

---

### Quan trọng nhất

```text
MEL → NV
```

Đây là lỗi nguy hiểm nhất.

---

## 8.7 McNemar Test

So sánh:

```text
Baseline

vs

Multimodal
```

Mục tiêu:

```text
p < 0.05
```

Ý nghĩa:

Metadata thực sự tạo khác biệt.

---

## 8.8 Bootstrap Confidence Interval

Phương pháp:

```text
1000 lần Bootstrap
```

Output:

```text
95% CI
```

Mục tiêu:

```text
≤ ±2.5%
```

---

## 8.9 Explainable AI

### Grad-CAM

Mục tiêu:

```text
Model nhìn đúng vùng tổn thương
```

Output:

```text
gradcam_samples.png
```

---

### SHAP

Mục tiêu:

```text
Metadata nào quan trọng nhất
```

Kỳ vọng:

```text
Age

>

Anatom Site

>

Sex
```

---

## 8.10 Output Files

```text
reports/

├── confusion_matrix.png

├── roc_curve_per_class.png

├── gradcam_samples.png

├── shap_feature_importance.png

└── evaluation_report.md
```