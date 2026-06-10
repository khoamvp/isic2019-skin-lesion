# ISIC 2019 — Multimodal Skin Lesion Classification
> **Hệ thống AI Phân loại Tổn thương Da liễu Đa phương thức**

**Tác giả:**  
**Phiên bản:** 1.0  
**Ngày tạo:**  
**Cập nhật lần cuối:**  

---

## PHẦN 1 — PROJECT BRIEF

**Tên dự án:** Multimodal Skin Lesion Classification System  
**Bộ dữ liệu:** ISIC 2019 (25,331 ảnh dermoscopy, 8 lớp bệnh lý)

**Bài toán:**  
Xây dựng hệ thống AI phân loại 8 loại bệnh lý da liễu từ ảnh dermoscopy kết hợp metadata lâm sàng (tuổi, giới tính, vị trí tổn thương), tập trung vào việc phân biệt chính xác ung thư hắc tố (Melanoma) với nốt ruồi lành tính (Nevi).

**Giá trị thực tế:**  
Ung thư hắc tố nếu phát hiện muộn có tỷ lệ tử vong cao, nhưng nếu phát hiện sớm thì tỷ lệ sống sót trên 95%. Hệ thống giúp bác sĩ da liễu sàng lọc nhanh hơn, giảm tỷ lệ bỏ sót ca ác tính, đặc biệt hữu ích ở các cơ sở thiếu chuyên gia.

**Người dùng mục tiêu:**  
Bác sĩ da liễu và kỹ thuật viên y tế tại phòng khám. Hệ thống đóng vai trò công cụ hỗ trợ quyết định lâm sàng (CDSS), không thay thế chẩn đoán của bác sĩ.

**Ngoài phạm vi (out-of-scope):**
- Không tích hợp với hệ thống HIS/EMR
- Không hỗ trợ ảnh ngoài định dạng dermoscopy (ảnh điện thoại thông thường không được kiểm định)
- Không dùng cho cơ sở thiếu kết nối internet ổn định
- Không thay thế chẩn đoán lâm sàng — chỉ là công cụ hỗ trợ (CDSS)
- Không áp dụng cho bệnh nhân nhi (dữ liệu ISIC 2019 chủ yếu người trưởng thành)

**Thể hiện trong portfolio:**
- Multimodal deep learning (CNN + MLP fusion)
- Xử lý class imbalance nặng trong y tế
- Explainable AI (Grad-CAM + SHAP)
- End-to-end: từ data pipeline đến deploy Streamlit
- Đánh giá theo tiêu chuẩn y khoa (Sensitivity, 95% CI, McNemar test)

---

## PHẦN 2 — DATA CARD

**2.1 — Nguồn dữ liệu:**  
ISIC 2019 Challenge Dataset, tổng hợp từ 3 nguồn: BCN_20000, HAM10000, MSK Dataset.  
Kaggle: `alifshahariar/isic-2019-dataset-full`

**2.2 — Quy mô:**  
25,331 ảnh dermoscopy chất lượng cao + 1 file CSV metadata lâm sàng

**2.3 — Cấu trúc CSV:**

| Cột | Kiểu | Mô tả |
|---|---|---|
| image | string | Tên file ảnh |
| MEL, NV, BCC, AK, BKL, DF, VASC, SCC | float | One-hot label 8 lớp |
| age_approx | float | Tuổi bệnh nhân, có NaN |
| sex | string | Giới tính, có NaN |
| anatom_site_general | string | Vị trí tổn thương, có NaN |
| lesion_id | string | Mã bệnh nhân (1 BN có nhiều ảnh) |

**2.4 — Phân phối nhãn:**

| Lớp | Số lượng | Tỷ lệ |
|---|---|---|
| NV (Nốt ruồi lành) | ~12,875 | ~51% |
| MEL (Ung thư hắc tố) | ~4,522 | ~18% |
| BCC | ~3,323 | ~13% |
| BKL | ~2,624 | ~10% |
| AK, DF, VASC, SCC | ~987 | ~8% còn lại |

**2.4b — EDA Insights lâm sàng:**

| Chiều phân tích | Insight |
|---|---|
| Tuổi vs Loại bệnh | MEL tập trung ở nhóm 45-65 tuổi, NV phổ biến nhất ở nhóm 20-40 tuổi |
| Vị trí vs Loại bệnh | Lưng/thân mình → MEL hoặc NV; Mặt → BCC và AK (tiền ung thư) |
| Giới tính vs Loại bệnh | MEL gặp nhiều ở nam hơn nữ; DF ngược lại, gặp nhiều ở nữ hơn |

> Các insight này là cơ sở để chứng minh metadata lâm sàng có giá trị thực sự khi kết hợp với ảnh.

**2.5 — Thách thức đã biết:**
- Class imbalance cực nặng: NV chiếm 51%
- Cặp MEL vs NV hình thái học rất giống nhau trên ảnh dermoscopy
- Metadata có NaN: age_approx ~8%, sex ~4%, anatom_site_general ~10% (ước tính, cần verify lại bằng df.isnull().sum())
- 1 bệnh nhân có nhiều ảnh → phải split theo lesion_id không phải image_id

**2.5b — Chất lượng ảnh:**
- Định dạng: JPEG
- Độ phân giải: đa dạng (từ ~600×450 đến ~1024×1024), cần resize về 224×224 khi đưa vào model
- Color space: RGB
- Nguồn máy chụp: dermoscope kỹ thuật số từ nhiều thiết bị khác nhau (BCN, HAM, MSK) → có thể có domain shift giữa các nguồn

**2.6 — Ràng buộc:**
- Dữ liệu đã ẩn danh hóa, chỉ dùng cho mục đích nghiên cứu và học thuật
- Không dùng cho chẩn đoán lâm sàng thực tế khi chưa được kiểm định y tế

---

## PHẦN 3 — TECHNICAL DESIGN

**3.1 — Kiến trúc mô hình:**

> **Lý do chọn ResNet50:** Được sử dụng rộng rãi trong tài liệu ISIC (baseline phổ biến nhất để so sánh), nhẹ hơn EfficientNet-B7, Grad-CAM hoạt động tốt hơn so với ViT, phù hợp với GPU free tier của Kaggle. Có thể upgrade lên EfficientNet-B3 ở iteration sau nếu kết quả chưa đạt.

```
Ảnh (224x224x3)                    Metadata (vector)
      ↓                                   ↓
ResNet50 pretrained                  MLP Branch
(freeze block1-3,                  Linear(n → 64)
 unfreeze layer4+fc)                BatchNorm1d(64)
      ↓                             ReLU + Dropout(0.3)
 2048 dim vector                    Linear(64 → 16)
      ↓                                   ↓
      └──────────── concat ──────────────┘
                       ↓
                  2064 dim vector
                       ↓
              Linear(2064 → 8)
                       ↓
                 Softmax (8 lớp)
```

**3.2 — Chiến lược xử lý imbalance:**
- Focal Loss + Class Weights (phạt nặng khi sai lớp hiếm)
- Threshold shifting: hạ ngưỡng MEL xuống 0.35 thay vì 0.5
- Augmentation mạnh cho lớp thiểu số

**3.3 — Data Split:**
- StratifiedGroupKFold 5-fold
- Group = lesion_id (tránh data leakage theo bệnh nhân)
- Stratify = label (giữ tỷ lệ 8 lớp đồng đều)
- Tập Test khóa từ đầu, không bao giờ dùng để tune model

**3.4 — Training config:**

| Thành phần | Lựa chọn |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| Scheduler | CosineAnnealingLR, T_max=20 |
| Batch size | 32 |
| Epochs | 30 (max) |
| Early stopping | patience=5, monitor=val Macro-F1 |
| Gradient clipping | max_norm=1.0 |
| Seed | 42 (torch, numpy, random, CUDA) |
| Môi trường train | Kaggle (GPU free) |

**3.5 — Explainability:**
- Grad-CAM: trỏ vào layer4[-1] của ResNet, kèm Drop-and-Gain Evaluation
- SHAP KernelExplainer: đo mức độ đóng góp từng feature metadata

**3.6 — Stack kỹ thuật:**

| Mục | Tool |
|---|---|
| Deep learning | PyTorch |
| Augmentation | Albumentations |
| Experiment tracking | W&B |
| Data versioning | DVC + Google Drive |
| App | Streamlit |
| Deploy | Streamlit Cloud |
| Repo | GitHub |

**3.7 — Baseline vs Multimodal:**
- Chạy song song 2 model cùng seed, cùng fold
- Baseline: ResNet50 chỉ nhận ảnh
- Multimodal: ResNet50 + MLP fusion
- So sánh bằng McNemar test, chỉ công nhận cải tiến khi p < 0.05

---

## PHẦN 4 — ENVIRONMENT SETUP

**4.1 — Cấu trúc thư mục:**

```
isic2019-skin-lesion/
├── data/                    ← KHÔNG commit (DVC quản lý)
│   ├── raw/                 ← ảnh gốc ISIC2019
│   └── processed/           ← CSV sau khi clean
├── notebooks/               ← chỉ nháp EDA, không chứa logic
├── src/
│   ├── preprocessing.py     ← encode metadata, transform
│   ├── dataset.py           ← Custom PyTorch Dataset
│   ├── model.py             ← Baseline + Multimodal model
│   ├── train.py             ← training loop
│   └── evaluate.py          ← metrics, McNemar, bootstrap
├── models/                  ← KHÔNG commit (DVC quản lý)
├── reports/
│   └── figures/             ← confusion matrix, plots
├── docs/
│   ├── project_plan.md      ← file này
│   └── experiment_log.md    ← ghi chép từng run
├── app.py                   ← Streamlit app
├── requirements.txt         ← full deps cho development
├── requirements_deploy.txt  ← deps tối giản cho Streamlit Cloud
├── .gitignore
├── .dvc/
└── README.md
```

**4.2 — .gitignore:**

```
data/
models/
venv/
__pycache__/
*.pth
*.pt
*.pkl
.env
.ipynb_checkpoints/
wandb/
```

**4.3 — Git LFS (quản lý file model nặng):**

```bash
# Cài Git LFS 1 lần duy nhất
git lfs install

# Track file .pth trước khi add
git lfs track "*.pth"

# Commit file .gitattributes do Git LFS tạo ra
git add .gitattributes
git commit -m "chore: configure git lfs for model weights"
```

> Bắt buộc chạy trước khi add bất kỳ file .pth nào.
> GitHub từ chối file > 100MB nếu không dùng Git LFS.

**4.4 — Requirements.txt:**

```
torch==2.1.0
torchvision==0.16.0
albumentations==1.3.1
streamlit==1.28.0
scikit-learn==1.3.2
pandas==2.1.0
numpy==1.24.0
opencv-python==4.8.0.76
grad-cam==1.4.8
shap==0.43.0
wandb==0.16.0
joblib==1.3.2
dvc==3.30.0
dvc-gdrive==2.20.0
```

**4.5 — Checklist setup:**

```
[ ] 1.  Tạo cấu trúc thư mục
[ ] 2.  Tạo .gitignore
[ ] 3.  git init + tạo repo GitHub + git push
[ ] 4.  Tạo venv, activate, pip install requirements.txt
[ ] 5.  git lfs install + git lfs track "*.pth"
[ ] 6.  dvc init
[ ] 7.  dvc remote add Google Drive
[ ] 8.  dvc add data/raw/ + dvc push
[ ] 9.  git add .dvc + commit + push
[ ] 10. Tạo tập sample 50 ảnh (6-7 ảnh/lớp) + CSV tương ứng
[ ] 11. Tạo project trên W&B, lưu API key vào .env
[ ] 12. Tạo Kaggle notebook, clone repo, mount Drive
[ ] 13. Tạo thư mục docs/, lưu project_plan.md
```

> Không được chuyển sang Phần 5 nếu chưa tick xong mục 1→10.

---

## PHẦN 5 — DATA PIPELINE SPEC

**5.1 — Tổng quan luồng dữ liệu:**

```
ISIC2019_train_metadata.csv
        ↓
   Load + Inspect
        ↓
  Xử lý null/NaN
        ↓
  Encode Metadata
        ↓
  StratifiedGroupKFold
        ↓
    ┌───┴───┐
  Train    Test (khóa)
    ↓
  Dataset + DataLoader
    ↓
  Augmentation (train only)
    ↓
  Normalize (tất cả)
    ↓
  Tensor → Model
```

**5.2 — Bước 1: Load + Inspect**

```
Input  : ISIC_2019_Training_Metadata.csv
         ISIC_2019_Training_GroundTruth.csv
Output : df_meta, df_gt đã merge

Kiểm tra:
- df.info()                    → kiểu dữ liệu từng cột
- df.isnull().sum()            → đếm NaN từng cột
- df['label'].value_counts()   → phân phối 8 lớp
- assert không có image_id trùng nhau
```

**5.3 — Bước 2: Xử lý null**

```
age_approx          → fillna(median), lưu median_age ra file
sex                 → fillna('unknown')
anatom_site_general → fillna('unknown')

Sau khi xử lý:
- assert df['age_approx'].isnull().sum() == 0
- assert df['sex'].isnull().sum() == 0
- assert df['anatom_site_general'].isnull().sum() == 0
```

**5.4 — Bước 3: Encode Metadata**

```
age_approx → StandardScaler → age_scaled
             lưu scaler ra models/scaler.pkl

sex + anatom_site_general → pd.get_dummies (One-Hot)
             lưu expected_cols ra models/feature_cols.pkl

Bỏ cột: age_approx, lesion_id (sau khi split)

Output: vector metadata shape (n_samples, n_features)
```

**5.5 — Bước 4: Data Split**

```
Phương pháp : StratifiedGroupKFold(n_splits=5)
Group        : lesion_id (tránh data leakage theo bệnh nhân)
Stratify     : label (giữ tỷ lệ 8 lớp đồng đều)
Tỷ lệ        : 80% train / 20% test (fold đầu tiên)

Sau khi split:
- reset_index(drop=True) cả 2 tập
- assert len(set(train_df['lesion_id']) &
         set(test_df['lesion_id'])) == 0
- assert test_df['label'].nunique() == 8
```

**5.6 — Bước 5: Dataset + DataLoader**

```
Custom Dataset (src/dataset.py):
- __init__   : nhận df + transform
- __len__    : len(df)
- __getitem__:
    1. đọc ảnh bằng cv2.imread
    2. cv2.cvtColor BGR→RGB       ← bắt buộc
    3. fallback nếu ảnh None: trả về ảnh đen np.zeros((224,224,3))
       + log warning với tên file để debug sau
    4. apply transform
    5. lấy metadata vector
    6. lấy label
    7. return img_tensor, meta_tensor, label

DataLoader:
- batch_size  = 32
- shuffle     = True (train) / False (val/test)
- num_workers = 0 (Windows) / 2 (Kaggle/Linux)
- Xử lý imbalance train: WeightedRandomSampler
  weight[i] = 1 / class_count[label[i]]
  Kết hợp với Focal Loss để tăng hiệu quả cho lớp thiểu số
```

**5.7 — Bước 6: Transform**

```
TRAIN transform (lớp phổ biến NV, MEL, BCC, BKL):
  A.Resize(224, 224)
  A.HorizontalFlip(p=0.5)
  A.RandomRotate90(p=0.5)
  A.ColorJitter(p=0.3)
  A.Normalize(ImageNet mean/std)
  ToTensorV2()

TRAIN transform bổ sung (lớp thiểu số AK, DF, VASC, SCC):
  Thêm vào trên:
  A.ShiftScaleRotate(p=0.5)
  A.ElasticTransform(p=0.3)
  A.GridDistortion(p=0.3)
  → Áp dụng thông qua WeightedRandomSampler hoặc conditional augment

VAL/TEST/INFERENCE (INFERENCE_TRANSFORM):
  A.Resize(224, 224)
  A.Normalize(ImageNet mean/std)
  ToTensorV2()

Lưu ý: INFERENCE_TRANSFORM định nghĩa 1 lần
        trong preprocessing.py, dùng chung
        cho dataset.py và app.py
```

**5.8 — Sanity Check trước khi train**

```
Chạy trên 50 ảnh mẫu, kiểm tra:
[ ] img_tensor.shape  == (32, 3, 224, 224)
[ ] meta_tensor.shape == (32, n_features)
[ ] label.shape       == (32,)
[ ] img_tensor.dtype  == torch.float32
[ ] meta_tensor.dtype == torch.float32
[ ] label.dtype       == torch.int64
[ ] img pixel range   ≈ [-2.5, 2.5] sau normalize
[ ] 8 lớp đều xuất hiện trong 1 epoch
```

**5.9 — Files cần lưu sau Pipeline:**

```
models/
├── scaler.pkl        ← StandardScaler đã fit
├── median_age.pkl    ← median tuổi từ train
└── feature_cols.pkl  ← danh sách cột one-hot đúng thứ tự
```

---

## PHẦN 6 — EXPERIMENT TRACKING PLAN

**6.1 — Công cụ: Weights & Biases (W&B)**

```
Project name: "isic2019-skin-lesion"
Mỗi lần train = 1 Run riêng biệt

Đặt tên run:
  - baseline_resnet50_fold1
  - multimodal_fold1_focal_loss
  - multimodal_fold1_threshold035

Khi mất kết nối internet (Kaggle):
  wandb.init(mode="offline")
  → Sync sau bằng: wandb sync ./wandb/offline-run-*
```

**6.2 — Log theo từng Epoch:**

```python
wandb.log({
    "train/loss"        : train_loss,
    "val/loss"          : val_loss,
    "val/accuracy"      : val_acc,
    "val/macro_f1"      : macro_f1,
    "val/macro_auc"     : macro_auc,
    "val/mel_recall"    : mel_recall,
    "val/mel_precision" : mel_precision,
    "train/learning_rate": current_lr,
    "epoch"             : epoch
})
```

**6.3 — Log cuối Training:**

```python
# Confusion matrix
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=true_labels,
    preds=pred_labels,
    class_names=LABEL_COLS
)})

# F1 từng lớp
wandb.log({f"test/{cls}_f1": f1_per_class[cls] for cls in LABEL_COLS})

# Grad-CAM samples
wandb.log({"gradcam_samples": [wandb.Image(img) for img in cam_samples]})
```

**6.4 — Checkpoint Strategy:**

```python
checkpoint = {
    'epoch'          : epoch,
    'model_state'    : model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'best_val_f1'    : best_val_f1,
    'config'         : CONFIG
}
# Lưu khi val Macro-F1 tốt hơn best
# Đẩy lên Google Drive qua DVC ngay sau đó
```

**6.5 — Experiment Comparison Table:**

| Run | Macro-F1 | Macro-AUC | MEL Recall | Threshold | p-value |
|---|---|---|---|---|---|
| Baseline (image only) | - | - | - | 0.5 | - |
| Multimodal | - | - | - | 0.5 | - |
| Multimodal + threshold | - | - | - | 0.35 | McNemar |

> Điều kiện công nhận: Delta Macro-F1 ≥ 3%, MEL Recall ≥ 85%, p < 0.05

**6.6 — Config:**

```python
CONFIG = {
    "model"        : "multimodal",
    "backbone"     : "resnet50",
    "lr"           : 1e-4,
    "batch_size"   : 32,
    "epochs"       : 30,
    "mel_threshold": 0.35,
    "n_folds"      : 5,
    "seed"         : 42,
    "loss"         : "focal",
    "img_size"     : 224
}
```

**6.7 — Experiment Log (docs/experiment_log.md):**

```markdown
## Run: <tên run>
**Date:** dd/mm/yyyy
**Config:** lr=?, batch=?, epoch=?, loss=?, threshold=?
**Result:**
- Macro-F1  : 0.xx
- Macro-AUC : 0.xx
- MEL Recall: 0.xx
**Observation:** (nhận xét ngắn)
**Next step:** (việc cần làm tiếp)
```

---

## PHẦN 7 — MILESTONE & DONE CRITERIA

**7.1 — Tổng quan:**

```
M0: Nền móng         ← bắt đầu từ đây
M1: Data Pipeline
M2: Baseline Model
M3: Multimodal Model
M4: Evaluation
M5: Streamlit App
M6: Deploy + Portfolio
```

**7.2 — Chi tiết:**

**M0 — Nền móng** | Thời gian: 1 ngày
```
[ ] Cấu trúc thư mục đúng chuẩn
[ ] .gitignore hoạt động, không commit data/models
[ ] DVC push data lên Google Drive thành công
[ ] Sample 50 ảnh đã có sẵn để test pipeline
[ ] W&B project đã tạo, API key đã lưu .env
[ ] Kaggle notebook clone repo được
```

**M1 — Data Pipeline** | Thời gian: 2-3 ngày
```
[ ] encode_metadata() chạy đúng, lưu scaler/median/cols
[ ] StratifiedGroupKFold không bị data leakage
[ ] assert giao tập train-test = rỗng
[ ] assert test có đủ 8 lớp
[ ] Sanity check batch shape đúng hết
[ ] Chạy thông trên 50 ảnh mẫu, không lỗi
```

**M2 — Baseline Model** | Thời gian: 2-3 ngày
```
[ ] src/model.py có BaselineModel
[ ] Train 1 fold trên Kaggle, W&B log đủ metrics
[ ] Có checkpoint best_baseline.pth
[ ] Ghi kết quả vào experiment_log.md
[ ] Macro-F1 > 0.60 (ngưỡng tối thiểu để tiếp tục)
```

**M3 — Multimodal Model** | Thời gian: 3-5 ngày
```
[ ] src/model.py có MultimodalNet
[ ] Focal Loss + Class Weights tích hợp
[ ] Train 5 folds trên Kaggle
[ ] Delta Macro-F1 ≥ 3% so với Baseline
[ ] MEL Recall ≥ 85% trên val
[ ] Có checkpoint best_multimodal.pth
```

**M4 — Evaluation** | Thời gian: 2-3 ngày
```
[ ] Chạy trên tập Test đã khóa từ M1
[ ] Macro-AUC ≥ 0.92
[ ] MEL Recall ≥ 85%
[ ] McNemar test p < 0.05
[ ] Bootstrapping 1000 lần, 95% CI ≤ ±2.5%
[ ] Grad-CAM heatmap hợp lý về mặt y khoa
[ ] SHAP cho thấy metadata có đóng góp thực sự
```

**M5 — Streamlit App** | Thời gian: 3-4 ngày
```
[ ] Upload ảnh + nhập metadata → ra kết quả
[ ] Hiển thị đủ 4 output: xác suất, heatmap, risk label, văn bản
[ ] @st.cache_resource hoạt động, không RAM leak
[ ] Missing metadata không crash app
[ ] Inference time < 3 giây
[ ] Chạy ổn định 10 ảnh liên tiếp không sập
```

**M6 — Deploy + Portfolio** | Thời gian: 1-2 ngày
```
[ ] App deploy trên Streamlit Cloud, link public
[ ] README.md tiếng Anh đầy đủ 5 mục
[ ] Mermaid diagram kiến trúc model trong README
[ ] GIF demo đính kèm đầu README
[ ] docs/ hoàn chỉnh tất cả phần
[ ] GitHub repo clean, commit message rõ ràng
[ ] Không có data/models nào bị commit nhầm
```

**7.3 — Tổng thời gian ước tính:**

```
M0 : 1 ngày
M1 : 2-3 ngày
M2 : 2-3 ngày
M3 : 3-5 ngày
M4 : 2-3 ngày
M5 : 3-4 ngày
M6 : 1-2 ngày
─────────────
Tổng: 14-21 ngày (part-time ~3-4h/ngày)
```

**7.4 — Timeline (điền ngày bắt đầu thực tế):**

| Milestone | Bắt đầu | Deadline | Buffer |
|---|---|---|---|
| M0 | __ / __ | __ / __ | +1 ngày |
| M1 | __ / __ | __ / __ | +1 ngày |
| M2 | __ / __ | __ / __ | +2 ngày |
| M3 | __ / __ | __ / __ | +2 ngày |
| M4 | __ / __ | __ / __ | +1 ngày |
| M5 | __ / __ | __ / __ | +2 ngày |
| M6 | __ / __ | __ / __ | +1 ngày |

**7.5 — Quy tắc:**
- Không chuyển Milestone tiếp khi chưa tick đủ Done Criteria
- Nếu M2 Macro-F1 < 0.60 thì dừng, debug pipeline trước
- Tập Test chỉ được mở duy nhất 1 lần ở M4

**7.6 — Kế hoạch khi kết quả không đạt:**

| Tình huống | Hành động |
|---|---|
| M2 Macro-F1 < 0.60 | Debug data pipeline, kiểm tra label mapping, thử lr=1e-3 |
| M3 MEL Recall < 0.85 sau 2 lần thử | Hạ threshold MEL xuống 0.30, tăng class weight MEL lên 3× |
| M3 Delta F1 < 3% | Xem xét thêm feature metadata (lesion size nếu có), thử EfficientNet-B3 |
| Kaggle GPU quota hết | Chuyển sang Google Colab free, giảm batch_size xuống 16 |
| M4 CI > ±2.5% | Tăng bootstrap lên 2000 lần, kiểm tra tập test có đủ đại diện 8 lớp không |

---

## PHẦN 8 — EVALUATION PLAN

**8.1 — Metrics tổng thể (Model-level):**

| Metric | Ngưỡng | Lý do |
|---|---|---|
| Macro-AUC-ROC | ≥ 0.92 | Đánh giá toàn diện 8 lớp, không bị NV chi phối |
| Macro-F1 | ≥ 0.75 | Trung bình F1 đồng đều 8 lớp, phạt lớp hiếm bị bỏ sót |
| Overall Accuracy | ≥ 0.85 | Metric cơ bản, không dùng một mình vì NV chiếm 51% |

**8.2 — Metrics lâm sàng (Clinical-level):**

| Metric | Ngưỡng | Lý do |
|---|---|---|
| MEL Recall (Sensitivity) | ≥ 0.85 | KPI số 1 — bỏ sót ung thư = nguy hiểm tính mạng |
| MEL Precision | càng cao càng tốt | Tỷ lệ báo nhầm, quá thấp thì bác sĩ mất tin tưởng |
| MEL F1-Score | ≥ 0.75 | Cân bằng Recall và Precision lớp ác tính |
| NV Specificity | ≥ 0.80 | Giảm báo động nhầm nốt ruồi lành thành ung thư |

**8.3 — Metrics thống kê y khoa:**

```
95% Confidence Interval (CI)
  Phương pháp : Bootstrapping 1000 lần trên tập Test
  Ngưỡng      : Biên độ dao động ≤ ±2.5%
  Lý do       : Chứng minh kết quả ổn định, không phải may mắn

McNemar Test
  So sánh     : Multimodal vs Baseline trên cùng tập Test
  Ngưỡng      : p < 0.05
  Lý do       : Chứng minh metadata thực sự có đóng góp
```

**8.4 — Metrics so sánh model:**

```
Delta Macro-F1 ≥ 3%
  Multimodal phải vượt Baseline ít nhất 3%
  mới công nhận cải tiến có giá trị

Per-class F1 từng lớp
  Xem lớp nào model còn yếu
  Đặc biệt chú ý: DF, VASC, SCC (số lượng ít nhất)
```

**8.5 — Metrics XAI:**

```
Grad-CAM Drop-and-Gain Evaluation
  Che vùng heatmap → xác suất phải giảm
  Nếu không giảm → model đang học vùng nền sai

SHAP Feature Importance
  Metadata nào đóng góp nhiều nhất
  Kỳ vọng: age > anatom_site > sex
```

**8.5b — Confusion Matrix Analysis:**

```
Sau khi có kết quả test, phân tích:
- Top-3 cặp bị nhầm nhiều nhất (ví dụ MEL→NV, BCC→BKL)
- Tính nhầm MEL sang NV: số lượng + % trên tổng MEL thực tế
  → Đây là False Negative nguy hiểm nhất về mặt lâm sàng
- So sánh confusion matrix Baseline vs Multimodal
  → Xem multimodal giảm nhầm MEL→NV được bao nhiêu %
```

**8.6 — Metrics hệ thống:**

```
Inference time   < 3 giây (từ upload đến ra kết quả)
Stability        : 10 ảnh liên tiếp không sập
RAM usage        : không tăng liên tục (không leak)
```

**8.7 — Thứ tự ưu tiên khi báo cáo:**

```
Bắt buộc đạt:
  1. MEL Recall ≥ 0.85
  2. Macro-AUC ≥ 0.92
  3. p < 0.05 (McNemar)
  4. 95% CI ≤ ±2.5%

Quan trọng:
  5. Macro-F1 ≥ 0.75
  6. NV Specificity ≥ 0.80
  7. Delta F1 ≥ 3% vs Baseline

Bổ sung:
  8. Per-class F1 từng lớp
  9. Grad-CAM Drop-and-Gain
  10. SHAP Feature Importance
```

**8.8 — File output của Evaluation (src/evaluate.py):**

```
reports/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve_per_class.png
│   ├── gradcam_samples.png
│   └── shap_feature_importance.png
└── evaluation_report.md      ← điền kết quả thực tế vào đây
```

---

## PHẦN 9 — MODEL OUTPUT SPEC

Hệ thống trả về 4 đầu ra lâm sàng cho mỗi ảnh:

**Output 1 — Bảng xác suất 8 lớp:**
```
Input  : 1 ảnh + metadata
Output : dict {class: probability} cho cả 8 lớp
         Ví dụ: {"MEL": 0.80, "NV": 0.15, "BCC": 0.05, ...}
Hiển thị: Bar chart ngang, sort theo xác suất giảm dần
Lưu ý  : Hiển thị Top-3 xác suất cao nhất nổi bật
         MEL threshold = 0.35 (không phải 0.5)
```

**Output 2 — Grad-CAM Heatmap:**
```
Input  : ảnh gốc + model
Output : ảnh heatmap đè lên ảnh gốc (màu đỏ = vùng quan trọng)
Target : layer4[-1] của ResNet50
Hiển thị: Song song ảnh gốc và ảnh heatmap
Lưu ý  : Denormalize tensor về [0,1] trước khi vẽ heatmap
```

**Output 3 — Risk Label (Phân tầng nguy cơ):**
```
🔴 NGUY CƠ CAO  : MEL prob > 0.35, hoặc BCC/SCC prob > 0.50
                  → "Cần can thiệp ngay, chỉ định sinh thiết"
🟡 THEO DÕI     : AK, BKL xác suất cao (> 0.50)
                  → "Theo dõi định kỳ 3-6 tháng"
🟢 LÀNH TÍNH    : NV, DF, VASC xác suất cao (> 0.50)
                  → "Lành tính, tái khám khi có thay đổi"
⚪ KHÔNG RÕ RÀNG: max prob < 0.40
                  → Xem Output 5b — không hiển thị risk label
```

**Output 4 — Văn bản lâm sàng tự động:**
```
Template:
"Bệnh nhân [giới tính], [tuổi] tuổi.
Tổn thương tại vùng [vị trí].
Kết quả phân tích AI: [lớp dự đoán] (xác suất [x]%).
Các chẩn đoán phân biệt: [Top-2 và Top-3].
Khuyến nghị: [nội dung từ Risk Label].

⚠ LƯU Ý: Kết quả này do hệ thống AI tạo ra, chỉ có giá trị hỗ trợ
quyết định lâm sàng. Không thay thế chẩn đoán của bác sĩ da liễu
có chuyên môn. Dữ liệu huấn luyện từ ISIC 2019 - chỉ dùng cho
mục đích nghiên cứu và học thuật."

Mục đích: bác sĩ copy paste thẳng vào hệ thống EMR
```

**Output 5b — Xử lý khi độ tin cậy thấp:**
```
Nếu max(probability) < 0.40:
  → Hiển thị cảnh báo: "Mô hình không đủ độ tin cậy để phân loại
    tổn thương này. Có thể do ảnh chất lượng thấp, góc chụp không
    chuẩn, hoặc tổn thương không điển hình."
  → Gợi ý: Chụp lại ảnh với dermoscope, đảm bảo không bị mờ/tối.
  → Không hiển thị Risk Label để tránh gây hiểu nhầm.
```

**Output 5 — Dashboard EDA:**
```
Mục đích : Giúp đội ngũ y tế hiểu đặc thù tập dữ liệu huấn luyện
Nội dung :
  - Lưới ảnh mẫu theo từng class bệnh lý (6-7 ảnh/lớp)
  - Biểu đồ phân bố số lượng mẫu 8 lớp (bar chart)
  - Biểu đồ phân bố tuổi theo từng lớp bệnh (box plot)
  - Biểu đồ tỷ lệ giới tính theo lớp bệnh (stacked bar)
  - Biểu đồ vị trí tổn thương theo lớp bệnh (heatmap)
Hiển thị : Tab riêng trong Streamlit app ("EDA Dashboard")
```

---

## PHẦN 10 — DEPLOYMENT PLAN

**10.1 — Kiến trúc deploy:**

```
GitHub repo (code)
      ↓
Streamlit Cloud (auto deploy khi push)
      ↓
Load model từ Google Drive URL
      ↓
User upload ảnh → inference → 4 outputs
```

**10.2 — Cấu hình Streamlit Cloud:**

```
Repository  : github.com/<username>/isic2019-skin-lesion
Branch      : main
Main file   : app.py
Python      : 3.10
```

**10.3 — Quản lý secrets (không hardcode):**

```toml
# .streamlit/secrets.toml (KHÔNG commit file này)
MODEL_URL   = "https://drive.google.com/..."
WANDB_KEY   = "..."

# Dùng trong app.py:
model_url = st.secrets["MODEL_URL"]
```

**10.4 — requirements_deploy.txt (tối giản, không có dev deps):**

```
torch==2.1.0
torchvision==0.16.0
albumentations==1.3.1
streamlit==1.28.0
scikit-learn==1.3.2
pandas==2.1.0
numpy==1.24.0
opencv-python-headless==4.8.0.76
grad-cam==1.4.8
joblib==1.3.2
```

> Dùng `opencv-python-headless` thay vì `opencv-python` trên server (không cần GUI).

**10.5 — Load model an toàn trong app.py:**

```python
@st.cache_resource
def load_model():
    # Download model từ Drive nếu chưa có
    model_path = Path("models/best_multimodal.pth")
    if not model_path.exists():
        download_from_drive(st.secrets["MODEL_URL"], model_path)

    model = MultimodalNet(meta_dim=META_DIM, num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
```

**10.6 — Checklist trước khi deploy:**

```
[ ] requirements_deploy.txt đã tối giản
[ ] .streamlit/secrets.toml đã thêm vào .gitignore
[ ] Model .pth load được từ Drive URL
[ ] App chạy local không lỗi
[ ] Missing metadata không crash
[ ] Inference time < 3 giây trên CPU
[ ] Low confidence case hiển thị đúng cảnh báo
[ ] Disclaimer AI hiển thị rõ ràng
[ ] Streamlit Cloud connect GitHub thành công
[ ] App live, link public hoạt động
```

**10.7 — Tối ưu model size cho Streamlit Cloud (RAM ~1GB):**

```
ResNet50 full precision (.pth) ≈ 100MB — có thể vừa đủ RAM
Nếu gặp MemoryError, áp dụng theo thứ tự:
  1. torch.jit.script(model) → lưu dạng TorchScript
  2. Dynamic quantization:
     torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
     → Giảm ~50% model size, giảm ~30% inference time trên CPU
  3. Nếu vẫn lỗi: upgrade Streamlit Cloud lên paid tier hoặc chuyển sang Hugging Face Spaces
```

**10.8 — Monitoring sau deploy:**

```
Streamlit Cloud built-in:
  - Xem log lỗi tại: share.streamlit.io → App → Logs
  - Uptime: Streamlit tự restart khi crash

Thủ công:
  - Ping app mỗi ngày để tránh bị sleep (free tier ngủ sau 7 ngày)
  - Log inference error ra st.exception() để user thấy thay vì crash im lặng

Rollback:
  - Giữ tag git: git tag -a v1.0 -m "first working deploy"
  - Khi cần rollback: Streamlit Cloud → Settings → chọn commit cũ
```

---

## PHẦN 11 — README SPEC

**11.1 — Cấu trúc README.md (tiếng Anh):**

```markdown
# Multimodal Skin Lesion Classification
> AI-powered clinical decision support for dermatologists

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Updated](https://img.shields.io/badge/updated-YYYY--MM-lightgrey)

[GIF Demo ở đây — đặt ngay đầu trang]

## 1. Medical Context
- Tại sao bài toán này quan trọng
- Melanoma early detection stats
- Hệ thống đóng vai trò CDSS, không thay thế bác sĩ

## 2. Dataset
- ISIC 2019, 25,331 images, 8 classes
- Class distribution table
- Challenge: imbalance + MEL vs NV confusion

## 3. Model Architecture
[Mermaid diagram ở đây]
- CNN Branch: ResNet50 pretrained
- MLP Branch: metadata → 16 dim
- Fusion: concat → classifier

## 4. Clinical Results
| Metric         | Value  |
|----------------|--------|
| Macro-AUC      | 0.xx   |
| MEL Recall     | 0.xx   |
| Macro-F1       | 0.xx   |
| 95% CI         | ±x.x%  |
| vs Baseline    | +x.x%  |

## 5. How to Run
# Clone + install
git clone ...
pip install -r requirements.txt

# Run app
streamlit run app.py

## 6. Limitations & Ethical Considerations
- Trained on ISIC 2019 (mostly Western/European skin tones) — may not generalize to all skin types
- Not validated in real clinical settings — for research/portfolio purposes only
- Class imbalance means performance on AK, DF, VASC, SCC is lower than MEL/NV
- Model may underperform on images taken with non-dermoscope cameras
- Always defer to a qualified dermatologist for diagnosis

## Live Demo
[Link Streamlit Cloud]
```

**11.2 — Mermaid diagram kiến trúc:**

```mermaid
graph TD
    A[Dermoscopy Image 224x224] --> B[ResNet50 pretrained]
    C[Metadata age/sex/location] --> D[MLP Branch]
    B --> E[2048-dim vector]
    D --> F[16-dim vector]
    E --> G[Concat 2064-dim]
    F --> G
    G --> H[Classifier Linear 8]
    H --> I[8-class Softmax]
    I --> J[Probability Table]
    I --> K[Grad-CAM Heatmap]
    I --> L[Risk Label]
    I --> M[Clinical Text]
```

**11.3 — GIF Demo:**

```
Tool ghi màn hình: ScreenToGif (Windows) hoặc LICEcap
Nội dung demo:
  1. Upload ảnh dermoscopy
  2. Nhập metadata (tuổi, giới tính, vị trí)
  3. Nhấn Analyze
  4. Hiển thị 4 outputs
  5. Thử ảnh MEL và ảnh NV để thấy sự khác biệt
Độ dài: 20-30 giây, không quá 5MB
```

---

## PHẦN 12 — RISK MANAGEMENT

**12.1 — Risk Register:**

| ID | Rủi ro | Xác suất | Mức độ | Mitigation |
|---|---|---|---|---|
| R01 | Kaggle GPU quota hết giữa chừng | Cao | Cao | Dùng Google Colab free làm backup; giảm batch_size xuống 16; chia train thành nhiều session |
| R02 | W&B mất kết nối khi train | Trung bình | Thấp | `wandb.init(mode="offline")`, sync sau; checkpoint local theo epoch |
| R03 | Model không converge (loss không giảm) | Trung bình | Cao | Sanity check trên 50 ảnh trước; kiểm tra learning rate với lr_finder; thử Adam với lr=1e-3 |
| R04 | MEL Recall không đạt 85% sau M3 | Trung bình | Cao | Hạ threshold xuống 0.25–0.30; tăng class weight MEL; thêm hard negative mining |
| R05 | Streamlit Cloud OOM (model quá lớn) | Trung bình | Trung bình | Quantize model; fallback Hugging Face Spaces (2GB RAM free) |
| R06 | DVC mất sync với Google Drive | Thấp | Trung bình | Giữ bản backup local; document đường dẫn Drive rõ ràng trong README |
| R07 | Dữ liệu ISIC 2019 bị xóa khỏi Kaggle | Thấp | Cao | Download và lưu local ngay từ đầu; DVC push lên Drive ngay M0 |
| R08 | Tiến độ bị trễ > 1 tuần | Trung bình | Trung bình | Cắt bỏ: EDA Dashboard (Output 5), SHAP (giữ Grad-CAM), 5-fold (giữ 1 fold) |

**12.2 — Thứ tự tính năng có thể cắt khi trễ (priority P1 → P3):**

```
P1 — Bắt buộc (không cắt):
  - Data pipeline đúng chuẩn
  - Baseline model + Multimodal model
  - MEL Recall + McNemar test
  - Streamlit app cơ bản (upload + kết quả)
  - README tiếng Anh + deploy

P2 — Quan trọng nhưng có thể đơn giản hóa:
  - 5-fold → giữ 1 fold (tiết kiệm 4x thời gian train)
  - SHAP → bỏ nếu trễ, chỉ giữ Grad-CAM
  - Bootstrapping CI → giảm xuống 500 lần thay vì 1000

P3 — Bỏ nếu cần:
  - EDA Dashboard (Output 5)
  - GIF demo (thay bằng screenshot tĩnh)
  - DCA (Decision Curve Analysis)
```

**12.3 — Checkpoint an toàn:**

```
Sau mỗi milestone hoàn thành:
  1. git tag -a mX-done -m "milestone X completed: <kết quả chính>"
  2. dvc push (đẩy data/model lên Drive)
  3. Ghi vào experiment_log.md
  4. Nghỉ ít nhất 1 ngày trước khi bắt đầu milestone tiếp
     → Tránh carry-over bug từ session mệt mỏi
```
