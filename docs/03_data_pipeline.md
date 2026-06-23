# PHẦN 5 — DATA PIPELINE

## 5.1 Mục tiêu

Data Pipeline chịu trách nhiệm:

* Load dữ liệu
* Kiểm tra chất lượng dữ liệu
* EDA
* Xử lý Missing Values
* Encode Metadata
* Tạo Fold
* Sinh dữ liệu đầu vào cho Model

Pipeline không thực hiện:

* Training
* Validation
* Metrics
* Inference

---

## 5.2 Workflow Tổng Thể

```text
ISIC Metadata CSV
↓
Load Metadata

ISIC GroundTruth CSV
↓
Load GroundTruth

Merge
↓
EDA
↓
Missing Value Handling
↓
Metadata Encoding
↓
Create Label Column
↓
Create Fold
↓
metadata_processed.csv
↓
dataset.py
↓
Image + Metadata Dataset
↓
train.py
```

---

## 5.3 Module Responsibility

### preprocessing.py

Chỉ thực hiện:

```text
Load Metadata CSV

Load GroundTruth CSV

Merge Data

EDA

Fill Missing Values

Encode Metadata

Create Fold

Save Artifacts
```

Không được chứa:

```text
OpenCV

PIL

Albumentations

Dataset Class

DataLoader

Training Logic
```

---

### dataset.py

Chịu trách nhiệm:

```text
Load Image

Albumentations

Tensor Conversion

Metadata Tensor

Label Tensor
```

---

### train.py

Chịu trách nhiệm:

```text
Fit StandardScaler

Training Loop

Validation Loop

Loss Function

Optimizer

Checkpoint

Early Stopping
```

---

### evaluate.py

Chịu trách nhiệm:

```text
Metrics

Confusion Matrix

ROC-AUC

Classification Report

McNemar Test

Bootstrap CI
```

---

## 5.4 Load Dữ Liệu

Input:

```text
ISIC_2019_Training_Metadata.csv

ISIC_2019_Training_GroundTruth.csv
```

Thực hiện:

```python
df_meta = pd.read_csv(...)

df_gt = pd.read_csv(...)
```

Merge:

```python
df = df_meta.merge(
    df_gt,
    on="image"
)
```

---

## 5.5 Tạo Label

GroundTruth là One-Hot Encoding.

Ví dụ:

```text
MEL=1
NV=0
BCC=0
...
```

Chuyển thành:

```python
df["label"]
```

Ví dụ:

```text
MEL

NV

BCC

...
```

---

## 5.6 Exploratory Data Analysis (EDA)

### Distribution

```python
df["label"].value_counts()
```

Output:

* Count
* Percentage

Biểu đồ:

```text
Bar Chart
```

---

### Missing Values

```python
df.isnull().sum()
```

Kiểm tra:

* age_approx
* sex
* anatom_site_general

---

### Age vs Disease

```python
sns.boxplot()
```

Mục tiêu:

Đánh giá:

```text
Age
↔
Disease Type
```

---

### Sex vs Disease

```python
pd.crosstab()
```

Output:

```text
Stacked Bar Chart
```

---

### Anatom Site vs Disease

```python
pd.crosstab()
```

Output:

```text
Heatmap
```

---

## 5.7 Missing Value Handling

### Age

```python
median_age = df["age_approx"].median()
```

Xử lý:

```python
fillna(median_age)
```

Lưu:

```text
models/median_age.pkl
```

---

### Sex

```python
fillna("unknown")
```

---

### Anatom Site

```python
fillna("unknown")
```

---

Kiểm tra:

```python
assert df.isnull().sum().sum() == 0
```

---

## 5.8 Metadata Encoding

### Age

Giữ nguyên giá trị số.

Ví dụ:

```text
52

37

68
```

Không StandardScaler tại đây.

---

### Sex

One-Hot Encoding

Ví dụ:

```text
male
female
unknown
```

↓

```text
sex_male
sex_female
sex_unknown
```

---

### Anatom Site

One-Hot Encoding

Ví dụ:

```text
back
head_neck
lower_extremity
unknown
```

↓

```text
site_back
site_head_neck
site_lower_extremity
...
```

---

### Feature Columns

Lưu:

```text
models/feature_cols.pkl
```

Mục tiêu:

Giữ đúng số chiều metadata khi:

* Validation
* Test
* Streamlit

---

## 5.9 Data Split

Sử dụng:

```python
StratifiedGroupKFold
```

Cấu hình:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

---

### Group

```python
groups=df["lesion_id"]
```

---

### Stratify

```python
y=df["label"]
```

---

### Kiểm tra bắt buộc

```python
assert len(
    set(train_df["lesion_id"])
    &
    set(val_df["lesion_id"])
) == 0
```

---

Mục tiêu:

Không để:

```text
Patient A
↓
Train

Patient A
↓
Validation
```

---

## 5.10 Output của preprocessing.py

Output chính:

```text
data/processed/
└── metadata_processed.csv
```

Artifacts:

```text
models/
├── median_age.pkl
└── feature_cols.pkl
```

Lưu ý:

```text
scaler.pkl

KHÔNG phải output chính
của preprocessing.py
```

---

## 5.11 Dataset Pipeline

dataset.py nhận:

```text
metadata_processed.csv

+

Image Path
```

Output:

```python
(
 image_tensor,
 metadata_tensor,
 label
)
```

---

## 5.12 Image Transform

### Train

```python
Resize(224,224)

HorizontalFlip

RandomRotate90

ColorJitter

Normalize

ToTensorV2
```

---

### Validation

```python
Resize(224,224)

Normalize

ToTensorV2
```

---

### Inference

Dùng cùng transform với Validation.

---

## 5.13 Sanity Check

Kiểm tra:

```python
img_tensor.shape
```

Kỳ vọng:

```text
(3,224,224)
```

---

```python
meta_tensor.shape
```

Kỳ vọng:

```text
(n_features,)
```

---

```python
label.dtype
```

Kỳ vọng:

```text
torch.int64
```

---

