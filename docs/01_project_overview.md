# ISIC 2019 — Multimodal Skin Lesion Classification

> Hệ thống AI Phân loại Tổn thương Da liễu Đa phương thức

**Version:** 2.0
**Created:** 11/06/2026
**Dataset:** ISIC 2019 Challenge Dataset
**Framework:** PyTorch
**Deployment:** Streamlit Cloud

---

# PHẦN 1 — PROJECT BRIEF

## 1.1 Tên dự án

**Multimodal Skin Lesion Classification System**

Hệ thống AI phân loại tổn thương da liễu từ ảnh dermoscopy kết hợp metadata lâm sàng.

---

## 1.2 Bài toán

Xây dựng mô hình Deep Learning có khả năng phân loại 8 loại bệnh lý da liễu từ:

* Ảnh dermoscopy
* Tuổi bệnh nhân
* Giới tính
* Vị trí tổn thương

Mục tiêu quan trọng nhất là:

> Phân biệt chính xác Melanoma (MEL) với Nevus (NV)

do đây là nhóm dễ nhầm lẫn nhất và có ý nghĩa lâm sàng cao nhất.

---

## 1.3 Giá trị thực tế

Melanoma là một dạng ung thư da có tỷ lệ tử vong cao nếu phát hiện muộn.

Nếu được phát hiện sớm:

* Tỷ lệ sống sót > 95%
* Điều trị đơn giản hơn
* Chi phí thấp hơn đáng kể

Hệ thống được xây dựng như một:

**Clinical Decision Support System (CDSS)**

giúp bác sĩ:

* Sàng lọc ban đầu
* Ưu tiên các ca nghi ngờ ác tính
* Giảm nguy cơ bỏ sót Melanoma

---

## 1.4 Người dùng mục tiêu

* Bác sĩ da liễu
* Kỹ thuật viên y tế
* Sinh viên y khoa
* Nhà nghiên cứu AI y tế

---

## 1.5 Out of Scope

Không thuộc phạm vi dự án:

* Không thay thế bác sĩ
* Không tích hợp HIS/EMR
* Không hỗ trợ ảnh điện thoại thông thường
* Không sử dụng cho chẩn đoán lâm sàng thực tế
* Không hỗ trợ dữ liệu ngoài ISIC 2019

---

## 1.6 Mục tiêu Portfolio

Dự án thể hiện:

* Computer Vision
* Multimodal Learning
* Medical AI
* Explainable AI
* MLOps cơ bản
* Streamlit Deployment

---

# PHẦN 2 — DATA CARD

## 2.1 Dataset

Tên:

**ISIC 2019 Challenge Dataset**

Nguồn:

* BCN_20000
* HAM10000
* MSK Dataset

Kaggle Dataset:

```text
alifshahariar/isic-2019-dataset-full
```

---

## 2.2 Quy mô dữ liệu

| Thành phần      | Số lượng |
| --------------- | -------- |
| Ảnh Dermoscopy  | 25,331   |
| Metadata CSV    | 1        |
| GroundTruth CSV | 1        |
| Số lớp          | 8        |

---

## 2.3 Metadata

| Cột                 | Ý nghĩa           |
| ------------------- | ----------------- |
| image               | tên file ảnh      |
| age_approx          | tuổi              |
| sex                 | giới tính         |
| anatom_site_general | vị trí tổn thương |
| lesion_id           | mã bệnh nhân      |

---

## 2.4 Các lớp bệnh

| Label | Mô tả                   |
| ----- | ----------------------- |
| MEL   | Melanoma                |
| NV    | Melanocytic Nevus       |
| BCC   | Basal Cell Carcinoma    |
| AK    | Actinic Keratosis       |
| BKL   | Benign Keratosis        |
| DF    | Dermatofibroma          |
| VASC  | Vascular Lesion         |
| SCC   | Squamous Cell Carcinoma |

---

## 2.5 Thách thức dữ liệu

### Class Imbalance

NV chiếm hơn 50% dữ liệu.

Các lớp:

* DF
* SCC
* VASC

có số lượng rất ít.

---

### Metadata Missing Values

Một số cột chứa NaN:

* age_approx
* sex
* anatom_site_general

---

### Patient Leakage

Một tổn thương có thể có nhiều ảnh.

Nếu chia ngẫu nhiên:

```text
Train:
Lesion A - Image 1

Validation:
Lession A - Image 2
```

sẽ gây Data Leakage.

Do đó:

```text
Group = lesion_id
```

là bắt buộc.

---

## 2.6 Quy định sử dụng

* Chỉ dùng cho học thuật
* Chỉ dùng cho nghiên cứu
* Không dùng để chẩn đoán lâm sàng thực tế

---
