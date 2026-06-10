import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

def process_and_split_real_data(
    meta_csv="data/raw/ISIC_2019_Training_Metadata.csv",
    gt_csv="data/raw/ISIC_2019_Training_GroundTruth.csv",
    image_dir="data/raw/train",
    output_dir="data/processed",
    models_dir="models"
):
    """
    Pipeline tiền xử lý dữ liệu thực tế ISIC 2019:
    1. Đồng bộ hóa bất đồng nhất tên file ảnh trên đĩa (_downsampled).
    2. Gộp Metadata lâm sàng với nhãn Ground Truth.
    3. Xử lý dữ liệu khuyết thiếu (NaN) chuẩn y khoa.
    4. Chuẩn hóa thang đo biến liên tục (Age).
    5. Mã hóa One-Hot biến định danh (Sex, Anatom Site).
    6. Chia Fold bằng StratifiedGroupKFold để chống rò rỉ dữ liệu bệnh nhân.
    """
    # Tạo các thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("--- Đang khởi động Data Pipeline với dữ liệu thực tế ISIC 2019 ---")
    
    # Kiểm tra sự tồn tại của file CSV gốc
    if not os.path.exists(meta_csv) or not os.path.exists(gt_csv):
        print(f"Lỗi: Không tìm thấy các file CSV gốc tại '{meta_csv}' hoặc '{gt_csv}'.")
        return
        
    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print(f"Lỗi: Thư mục ảnh mồi '{image_dir}' trống hoặc không tồn tại.")
        return

    # --- BƯỚC 1: XỬ LÝ BẤT ĐỒNG NHẤT TÊN ẢNH VÀ QUÉT Ổ ĐĨA ---
    print("--> Bước 1: Đang quét thư mục ảnh để xử lý khớp tên file thực tế...")
    image_files = os.listdir(image_dir)
    id_to_filename = {}
    for filename in image_files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):

            base_id = filename.split('.')[0]
            id_to_filename[base_id] = filename

    print(f"    Tìm thấy {len(id_to_filename)} ảnh mồi hợp lệ trên đĩa.")

    # --- BƯỚC 2: ĐỌC VÀ GỘP (MERGE) DỮ LIỆU ---
    print("--> Bước 2: Đang đọc và gộp file Metadata với GroundTruth...")
    df_meta = pd.read_csv(meta_csv)
    df_gt = pd.read_csv(gt_csv)
    df = pd.merge(df_meta, df_gt, on='image')
    
    # Ánh xạ tên file thực tế vào bảng dữ liệu dựa trên cột 'image'
    df['actual_image_filename'] = df['image'].map(id_to_filename)
    
    # Lọc bỏ các dòng trong CSV mà trên đĩa local hiện tại chưa có ảnh mồi tương ứng
    initial_count = len(df)
    df = df.dropna(subset=['actual_image_filename']).reset_index(drop=True)
    print(f"    Đã khớp thành công {len(df)} / {initial_count} dòng dữ liệu dựa trên số ảnh mồi đang có.")

    if len(df) == 0:
        print(" Lỗi: Không có ảnh mồi nào khớp với ID trong file CSV. Hãy kiểm tra lại tên ảnh mồi của bạn!")
        return

    # Trích xuất nhãn số (0-7) từ 8 cột ma trận one-hot phục vụ Stratified split
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    df['label'] = np.argmax(df[classes].values, axis=1)
    
    # --- BƯỚC 3: XỬ LÝ DỮ LIỆU KHUYẾT THIẾU (NaN) ---
    print("--> Bước 3: Đang xử lý các ca khuyết thiếu dữ liệu (NaN)...")
    median_age = df['age_approx'].median()
    # Nếu chạy ít ảnh quá không tính được median, đặt mặc định là 50.0
    if pd.isna(median_age): 
        median_age = 50.0
        
    df['age_approx'] = df['age_approx'].fillna(median_age)
    df['sex'] = df['sex'].fillna('unknown')
    df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')
    
    # Đảm bảo tuyệt đối không còn NaN 
    assert df['age_approx'].isnull().sum() == 0, "Lỗi logic: Vẫn còn NaN ở cột tuổi!"
    assert df['sex'].isnull().sum() == 0, "Lỗi logic: Vẫn còn NaN ở cột giới tính!"
    assert df['anatom_site_general'].isnull().sum() == 0, "Lỗi logic: Vẫn còn NaN ở cột vị trí tổn thương!"
    
    # Lưu giá trị median tuổi làm cấu hình cố định cho App Streamlit sau này
    joblib.dump(median_age, os.path.join(models_dir, "median_age.pkl"))
    
    # --- BƯỚC 4: CHUẨN HÓA THANG ĐO BIẾN LIÊN TỤC (AGE) ---
    print("--> Bước 4: Đang chuẩn hóa biến liên tục (Tuổi bệnh nhân)...")
    scaler = StandardScaler()
    # Fit và transform cột tuổi
    if len(df) > 1:
        df['age_scaled'] = scaler.fit_transform(df[['age_approx']])
    else:
        df['age_scaled'] = 0.0 # Phòng thủ nếu chỉ test duy nhất 1 ảnh
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    # --- BƯỚC 5: MÃ HÓA ONE-HOT BIẾN ĐỊNH DANH LÂM SÀNG ---
    print("--> Bước 5: Đang mã hóa One-Hot Encoding cho dữ liệu chữ...")
    expected_sex_cols = ['sex_female', 'sex_male', 'sex_unknown']
    expected_site_cols = ['site_anterior torso', 'site_head/neck', 'site_lower extremity', 
                          'site_palms/soles', 'site_posterior torso', 'site_unknown', 'site_upper extremity']
    
    df_sex_encoded = pd.get_dummies(df['sex'], prefix='sex', dtype=float)
    df_site_encoded = pd.get_dummies(df['anatom_site_general'], prefix='site', dtype=float)
    
    # Ép cấu hình cột theo đúng thiết kế Spec để khi Inference trên App không bị lệch số chiều
    df_sex_encoded = df_sex_encoded.reindex(columns=expected_sex_cols, fill_value=0.0)
    df_site_encoded = df_site_encoded.reindex(columns=expected_site_cols, fill_value=0.0)
    
    meta_features = ['age_scaled'] + expected_sex_cols + expected_site_cols
    joblib.dump(meta_features, os.path.join(models_dir, "feature_cols.pkl"))
    
    # Ghép các đặc trưng số hóa vào bảng DataFrame tổng
    df = pd.concat([df, df_sex_encoded, df_site_encoded], axis=1)
    
    # --- BƯỚC 6: CHIA FOLD CHỐNG RÒ RỈ DỮ LIỆU BỆNH NHÂN ---
    print("--> Bước 6: Đang thực hiện chia Fold bằng StratifiedGroupKFold...")
    df['fold'] = -1
    
    # Nếu số lượng ảnh mồi quá ít (nhỏ hơn 5), gán tạm fold=0 thay vì chia nhóm để tránh crash
    if len(df) >= 5 and df['lesion_id'].nunique() >= 5:
        sgkf = StratifiedGroupKFold(n_splits=5)
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df['label'], groups=df['lesion_id'])):
            df.loc[val_idx, 'fold'] = fold_idx
        
        # --- SANITY CHECK ĐỘ AN TOÀN DỮ LIỆU ---
        print("---  Đang chạy Sanity Check kiểm tra lỗi hệ thống ---")
        train_df = df[df['fold'] != 0]
        test_df = df[df['fold'] == 0]
        
        # Kiểm tra xem có bệnh nhân nào xuất hiện ở cả 2 tập không
        common_patients = set(train_df['lesion_id']).intersection(set(test_df['lesion_id']))
        print(f"    Số lượng bệnh nhân bị trùng lặp giữa Train và Test: {len(common_patients)}")
        assert len(common_patients) == 0, "CẢNH BÁO NGUY HIỂM: Rò rỉ dữ liệu bệnh nhân xuất hiện!"
        print("    -> Kết quả check rò rỉ: [ĐẠT] Code chia fold chống leak hoàn hảo.")
    else:
        df['fold'] = 0
        print("   Cảnh báo: Số lượng mẫu mồi quá ít, hệ thống tự động gán toàn bộ vào Fold 0 để test.")
    
    # Xuất file kết quả sạch ra thư mục processed
    output_path = os.path.join(output_dir, "metadata_processed.csv")
    df.to_csv(output_path, index=False)
    print(f"\ [THÀNH CÔNG] Toàn bộ dữ liệu sạch đã được xuất bản tại: {output_path}")

if __name__ == "__main__":
    process_and_split_real_data()