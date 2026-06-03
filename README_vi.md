# 🏠 House Rent Prediction

Dự án Machine Learning dự đoán **giá thuê nhà/căn hộ (Rent)** từ các đặc trưng bất động sản bằng các mô hình hồi quy trong thư viện **scikit-learn**.

Toàn bộ quy trình từ khám phá dữ liệu (EDA), tiền xử lý dữ liệu, feature engineering, mã hóa biến phân loại, xây dựng pipeline, tối ưu siêu tham số bằng **GridSearchCV** đến đánh giá mô hình đều được thực hiện trong notebook:

```text
House_Rent_Prediction_final.ipynb
```

---

# 📌 Mục tiêu dự án

Mục tiêu của dự án là xây dựng một mô hình hồi quy có khả năng dự đoán giá thuê nhà (`Rent`) dựa trên các đặc trưng như:

* Số phòng ngủ (`BHK`)
* Diện tích (`Size`)
* Tầng hiện tại và tổng số tầng (`Floor`)
* Loại khu vực (`Area Type`)
* Khu vực chi tiết (`Area Locality`)
* Thành phố (`City`)
* Tình trạng nội thất (`Furnishing Status`)
* Đối tượng thuê phù hợp (`Tenant Preferred`)
* Số phòng tắm (`Bathroom`)
* Người liên hệ (`Point of Contact`)
* Thời điểm đăng tin (`Posted On`)

Ngoài việc xây dựng mô hình dự đoán, dự án còn tập trung vào việc xử lý các vấn đề thường gặp trong dữ liệu thực tế như:

* Phân phối lệch phải của biến mục tiêu `Rent`
* Dữ liệu phân loại có cardinality cao (`Area Locality`)
* Đảm bảo không xảy ra data leakage trong quá trình tiền xử lý

---

# 📂 Dataset

**Tên file dữ liệu**

```text
House_Rent_Dataset.csv
```

**Nguồn dataset**

Kaggle: <https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>

## Thông tin tổng quan

| Thuộc tính                             | Giá trị |
| -------------------------------------- | ------- |
| Số dòng                                | 4746    |
| Số cột                                 | 12      |
| Biến mục tiêu                          | Rent    |
| Số giá trị khác nhau của Area Locality | 2235    |

Một số thành phố xuất hiện trong dữ liệu:

* Mumbai
* Chennai
* Bangalore
* Hyderabad
* Delhi
* Kolkata

## Các cột trong dataset

| Cột               | Ý nghĩa             |
| ----------------- | ------------------- |
| Posted On         | Ngày đăng tin       |
| BHK               | Số phòng ngủ        |
| Rent              | Giá thuê            |
| Size              | Diện tích           |
| Floor             | Thông tin tầng      |
| Area Type         | Loại khu vực        |
| Area Locality     | Khu vực chi tiết    |
| City              | Thành phố           |
| Furnishing Status | Tình trạng nội thất |
| Tenant Preferred  | Đối tượng thuê      |
| Bathroom          | Số phòng tắm        |
| Point of Contact  | Người liên hệ       |

---

# ⚙️ Pipeline Machine Learning

Dự án được triển khai theo pipeline Machine Learning đầy đủ:

```text
Dataset
   ↓
EDA
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
GridSearchCV
   ↓
Model Evaluation
```

## Các bước thực hiện

### 1. Đọc dữ liệu

Đọc dữ liệu từ file CSV.

### 2. Khám phá dữ liệu (EDA)

* Kiểm tra shape
* Kiểm tra kiểu dữ liệu
* Kiểm tra missing values
* Kiểm tra duplicate values
* Phân tích phân phối của biến `Rent`
* Trực quan hóa mối quan hệ giữa các đặc trưng và giá thuê

### 3. Feature Engineering

#### Xử lý cột `Posted On`

Chuyển sang kiểu datetime và tách thành:

* Posted_Month
* Posted_Day
* Posted_DayOfWeek

#### Xử lý cột `Floor`

Tách thành:

* Floor_Level
* Total_Floors

Ví dụ:

```text
3 out of 5
```

sẽ trở thành:

```text
Floor_Level = 3
Total_Floors = 5
```

---

# 🔧 Feature Engineering nâng cao

## 1. Log Transform cho Rent

Biến mục tiêu `Rent` có phân phối lệch phải mạnh và chứa nhiều outlier.

Để giảm ảnh hưởng của các giá trị cực lớn, mô hình được huấn luyện trên:

```python
y = np.log1p(data["Rent"])
```

Sau khi dự đoán:

```python
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)
```

### Lợi ích

* Giảm ảnh hưởng của outlier
* Giúp mô hình học ổn định hơn
* Dễ tối ưu hơn đối với dữ liệu giá thuê

---

## 2. Frequency Encoding cho Area Locality

Cột:

```text
Area Locality
```

có tới:

```text
2235 giá trị khác nhau
```

Nếu sử dụng One-Hot Encoding trực tiếp sẽ làm tăng số chiều dữ liệu rất lớn.

Thay vào đó sử dụng Frequency Encoding:

```python
area_locality_freq_map = X_train["Area Locality"].value_counts(normalize=True)
```

Tạo thêm đặc trưng:

```python
Area_Locality_Freq
```

### Ưu điểm

* Không làm tăng số chiều dữ liệu
* Giảm nguy cơ overfitting
* Tiết kiệm bộ nhớ
* Vẫn giữ lại thông tin vị trí

### Chống Data Leakage

Frequency map chỉ được tính trên tập train:

```python
area_locality_freq_map = X_train["Area Locality"].value_counts(normalize=True)

X_train["Area_Locality_Freq"] = X_train["Area Locality"].map(area_locality_freq_map)

X_test["Area_Locality_Freq"] = (
    X_test["Area Locality"]
    .map(area_locality_freq_map)
    .fillna(0)
)
```

---

# 📊 Đặc trưng sử dụng

## Numeric Features

* BHK
* Size
* Bathroom
* Posted_Month
* Posted_Day
* Posted_DayOfWeek
* Floor_Level
* Total_Floors
* Area_Locality_Freq

## Categorical Features

* Area Type
* City
* Furnishing Status
* Tenant Preferred
* Point of Contact

---

# 🤖 Các mô hình được thử nghiệm

Dự án so sánh 4 mô hình hồi quy:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Tất cả các mô hình đều được tối ưu bằng:

```python
GridSearchCV(
    cv=5,
    scoring="r2",
    n_jobs=-1
)
```

---

# 📈 Đánh giá mô hình

Các metric được tính trên giá trị `Rent` gốc sau khi hoàn nguyên từ log:

* MAE
* MSE
* RMSE
* R² Score

## Ý nghĩa

| Metric | Ý nghĩa                                    |
| ------ | ------------------------------------------ |
| MAE    | Sai số tuyệt đối trung bình                |
| MSE    | Sai số bình phương trung bình              |
| RMSE   | Sai số cùng đơn vị với Rent                |
| R²     | Khả năng giải thích phương sai của mô hình |

---

# 🏆 Kết quả thực nghiệm

| Model                       | Best CV R² (Log Rent) |  Test MAE | Test RMSE | Test R² |
| --------------------------- | --------------------: | --------: | --------: | ------: |
| Linear Regression           |                0.8120 | 11,396.86 | 31,826.16 |  0.7533 |
| Gradient Boosting Regressor |                0.8322 | 11,551.20 | 37,284.89 |  0.6614 |
| Random Forest Regressor     |                0.8254 | 11,364.49 | 40,178.34 |  0.6068 |
| Decision Tree Regressor     |                0.7763 | 12,808.13 | 40,675.65 |  0.5970 |

---

# 🥇 Mô hình tốt nhất

Mô hình tốt nhất theo kết quả trên tập test:

## Linear Regression

### Tham số tốt nhất

```python
{'regressor__fit_intercept': True}
```

### Kết quả cuối cùng

| Metric |          Giá trị |
| ------ | ---------------: |
| MAE    |        11,396.86 |
| MSE    | 1,012,904,643.11 |
| RMSE   |        31,826.16 |
| R²     |           0.7533 |

Mặc dù Gradient Boosting đạt điểm Cross Validation cao hơn trên thang log, nhưng Linear Regression cho kết quả tốt nhất khi đánh giá trên giá trị giá thuê thực tế.

---

# 📁 Cấu trúc project

```text
.
├── House_Rent_Prediction_final.ipynb
├── House_Rent_Dataset.csv
├── README.md
└── model_comparison_results.csv
```

---

# 🚀 Hướng dẫn chạy

## Chạy trên Google Colab

```python
from google.colab import drive

drive.mount('/content/drive')

data = pd.read_csv(
    "/content/drive/MyDrive/ML/BTL/Datasets/House_Rent_Dataset.csv"
)
```

### Các bước

1. Upload notebook lên Google Colab.
2. Đặt file CSV đúng vị trí trong Google Drive.
3. Chạy notebook từ đầu đến cuối.

---

## Chạy trên máy local

Sửa cell đọc dữ liệu:

```python
import pandas as pd

data = pd.read_csv("House_Rent_Dataset.csv")
```

Sau đó chạy toàn bộ notebook bằng:

* Jupyter Notebook
* VS Code
* Jupyter Lab

---

# 📦 Thư viện sử dụng

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Cài đặt:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

# ✅ Ưu điểm của dự án

* GridSearchCV tối ưu siêu tham số
* So sánh nhiều mô hình hồi quy
* Đánh giá trên giá trị Rent thực tế

---

# ⚠️ Hạn chế hiện tại


* Chưa thử các mô hình boosting nâng cao
* Chưa có hệ thống dự đoán thời gian thực

---

# 🔮 Hướng phát triển

Trong tương lai có thể:

* Thêm XGBoost
* Thêm LightGBM
* Thêm CatBoost
* Lưu mô hình bằng Joblib
* Xây dựng giao diện Streamlit
* Thêm SHAP để giải thích mô hình
* Tối ưu xử lý outlier

---

# 👨‍💻 Tác giả

```text
Author : <Nguyen Thi Thao My>
Project: House Rent Prediction
```

---

# 📜 License

Dự án có thể sử dụng giấy phép:

```text
MIT License
```
