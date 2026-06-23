# PHẦN 9 — MODEL OUTPUT SPECIFICATION

## 9.1 Mục Tiêu

Định nghĩa chuẩn đầu ra của mô hình để:

* Training
* Evaluation
* Streamlit Inference
* API tương lai

luôn sử dụng cùng định dạng.

---

## 9.2 Model Input

### Image

Kiểu dữ liệu:

```python
torch.FloatTensor
```

Shape:

```python
(3, 224, 224)
```

---

### Metadata

Kiểu dữ liệu:

```python
torch.FloatTensor
```

Shape:

```python
(n_features,)
```

Ví dụ:

```text
age

sex_male
sex_female
sex_unknown

site_back
site_head_neck
...
```

---

## 9.3 Model Output

Output gốc:

```python
logits
```

Shape:

```python
(batch_size, 8)
```

Ví dụ:

```python
[
  2.31,
  0.82,
 -1.25,
 ...
]
```

---

## 9.4 Probability

Sau Softmax:

```python
probabilities
```

Ví dụ:

```python
{
 "MEL": 0.74,
 "NV": 0.15,
 "BCC": 0.03,
 ...
}
```

Tổng:

```python
sum(probabilities) = 1
```

---

## 9.5 Predicted Class

```python
pred_class = argmax(probabilities)
```

Ví dụ:

```text
MEL
```

---

## 9.6 Clinical Risk Group

### High Risk

```text
MEL

SCC

BCC
```

Color: 🔴 Đỏ

---

### Medium Risk

```text
AK

BKL
```

Color: 🟡 Vàng

---

### Low Risk

```text
NV

DF

VASC
```

Color: 🟢 Xanh

---

Output:

```text
High Risk (🔴)
```

---

## 9.7 Confidence Score

### High Confidence

```text
Probability ≥ 0.80
```

---

### Medium Confidence

```text
0.60 ≤ Probability < 0.80
```

---

### Low Confidence

```text
Probability < 0.60
```

---

## 9.8 Streamlit Output Example

```text
Prediction:
Melanoma (MEL) — 74.2%

Top 3 Differential Diagnoses:

| Rank | Disease | Probability | Risk        | Confidence |
|------|---------|-------------|-------------|------------|
| 1    | MEL     | 74.2%       | 🔴 High     | Medium     |
| 2    | NV      | 15.0%       | 🟢 Low      | Low        |
| 3    | BCC     | 3.0%        | 🔴 High     | Low        |
```

---

## 9.9 JSON Output Format

```json
{
  "prediction": "MEL",
  "probability": 0.742,
  "risk": "High",
  "risk_color": "red",
  "confidence": "Medium",
  "top3": [
    {"class": "MEL", "probability": 0.742, "risk": "High", "risk_color": "red",   "confidence": "Medium"},
    {"class": "NV",  "probability": 0.150, "risk": "Low",  "risk_color": "green", "confidence": "Low"},
    {"class": "BCC", "probability": 0.030, "risk": "High", "risk_color": "red",   "confidence": "Low"}
  ]
}
```

---

## 9.10 Explainability Output

### Grad-CAM

Output:

```text
heatmap.png
```

---

### SHAP

Output:

```text
metadata_importance.png
```

---

## 9.11 Error Handling

### Missing Age

```text
NaN
↓
median_age.pkl
↓
Fill
```

---

### Missing Sex

```text
Unknown
```

---

### Missing Anatom Site

```text
Unknown
```

---

Ứng dụng không được crash khi metadata bị thiếu.

---

# PHẦN 10 — DEPLOYMENT PLAN

## 10.1 Mục Tiêu

Triển khai hệ thống trên:

```text
Streamlit Cloud
```

để người dùng có thể sử dụng trực tiếp qua trình duyệt.

---

## 10.2 Deployment Architecture

```text
User
↓
Streamlit UI
↓
Model Inference
↓
Prediction
↓
Grad-CAM
↓
Clinical Report
```

---

## 10.3 Files Cần Deploy

```text
app.py

best_model.pth

median_age.pkl

feature_cols.pkl
```

---

## 10.4 Không Deploy

Không đưa lên:

```text
Dataset Images

Raw CSV

Notebook

W&B Cache

Checkpoint Cũ
```

---

## 10.5 User Workflow

```text
Upload Image
↓
Nhập Metadata
↓
Predict
↓
Result
```

---

## 10.6 Input Form

### Image Upload

```text
jpg

jpeg

png
```

---

### Age

```text
Optional
```

Nếu bỏ trống:

```text
median_age.pkl
```

được sử dụng.

---

### Sex

```text
Male

Female

Unknown
```

---

### Anatom Site

```text
Back

Chest

Head/Neck

Upper Extremity

Lower Extremity

Unknown
```

---

## 10.7 Prediction Screen

Hiển thị:

### Patient Metadata

Ví dụ:

```text
Age: 45
Sex: Male
Anatom Site: Back
```

---

### Predicted Disease

Ví dụ:

```text
Melanoma
```

---

### Probability

Ví dụ:

```text
74.2%
```

---

### Risk Level

Ví dụ:

```text
High Risk (🔴)
```

---

### Confidence Level

Ví dụ:

```text
Medium Confidence
```

---

## 10.8 Probability Table

| Disease | Probability |
| ------- | ----------- |
| MEL     | 74.2%       |
| NV      | 15.0%       |
| BCC     | 3.0%        |
| ...     | ...         |

---

## 10.9 Grad-CAM Visualization

Hiển thị:

```text
Original Image

+

Heatmap Overlay
```

---

## 10.10 Clinical Report

Ví dụ:

```text
=== CLINICAL REPORT ===

Patient Information:
- Age: 45
- Sex: Male
- Anatom Site: Back

AI Diagnosis:
Predicted Class: Melanoma (MEL)
Probability: 74.2%
Risk Level: High (🔴)
Confidence: Medium

Top 3 Differential Diagnoses:
1. MEL — 74.2% — Risk: High
2. NV  — 15.0% — Risk: Low
3. BCC — 3.0%  — Risk: High

Recommendation:
Consult dermatologist for further evaluation and possible biopsy.
```

---

## 10.11 Performance Requirements

### Inference Time

```text
< 3 seconds
```

---

### Startup Time

```text
< 30 seconds
```

---

### Memory

```text
< 2 GB
```

---

## 10.12 Streamlit Project Structure

```text
app.py

src/
├── model.py
├── dataset.py
└── utils.py

models/
├── best_model.pth
├── median_age.pkl
└── feature_cols.pkl
```

---

## 10.13 Deployment Checklist

```text
[ ] best_model.pth tồn tại

[ ] median_age.pkl tồn tại

[ ] feature_cols.pkl tồn tại

[ ] requirements_deploy.txt chính xác

[ ] app.py chạy local

[ ] Streamlit Cloud hoạt động

[ ] Grad-CAM hoạt động

[ ] Không lỗi metadata thiếu

[ ] Public URL hoạt động
```

---

## 10.14 Future Upgrade

Có thể mở rộng:

```text
REST API

Docker

FastAPI

AWS

Azure

GCP
```

Nhưng không nằm trong phạm vi phiên bản hiện tại.