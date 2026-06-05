# House Rent Prediction

Dự án Machine Learning dự đoán giá thuê nhà và căn hộ (`Rent`) bằng các mô hình hồi quy trong `scikit-learn`.

Quy trình chính được triển khai trong [House_Rent_Prediction_ML.ipynb](./House_Rent_Prediction_ML.ipynb), gồm:

- Giới thiệu bài toán hồi quy
- Khám phá dữ liệu (EDA)
- Kiểm tra missing values và dữ liệu trùng lặp
- Tạo đặc trưng
- Tiền xử lý bằng `Pipeline` và `ColumnTransformer`
- Tối ưu siêu tham số bằng `GridSearchCV`
- Đánh giá mô hình trên thang giá thuê gốc
- Trực quan hóa kết quả và kết luận

## Mục Tiêu Dự Án

Mục tiêu là dự đoán giá thuê từ các đặc trưng bất động sản có cấu trúc như:

- `BHK`
- `Size`
- `Floor`
- `Area Type`
- `Area Locality`
- `City`
- `Furnishing Status`
- `Tenant Preferred`
- `Bathroom`

Notebook đi theo quy trình học có giám sát cho bài toán hồi quy: kiểm tra dữ liệu, biến đổi các cột thô thành đặc trưng phù hợp, huấn luyện nhiều mô hình, so sánh metric và chọn mô hình tốt nhất.

## Dataset

Tên file dữ liệu:

```text
House_Rent_Dataset.csv
```

Nguồn dữ liệu:

Kaggle: <https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>

### Các Cột Gốc

- `Posted On`
- `BHK`
- `Rent`
- `Size`
- `Floor`
- `Area Type`
- `Area Locality`
- `City`
- `Furnishing Status`
- `Tenant Preferred`
- `Bathroom`
- `Point of Contact`

## Cách Mô Hình Hóa Hiện Tại

### Biến Đổi Biến Mục Tiêu

Notebook huấn luyện mô hình trên:

```python
y = np.log1p(data["Rent"])
```

Cách này giúp giảm ảnh hưởng của outlier giá thuê cao và phân phối lệch phải mạnh của `Rent`. Sau khi dự đoán, kết quả được chuyển về thang giá thuê gốc bằng `np.expm1(...)` trước khi đánh giá.

### Các Cột Bị Loại Bỏ Trước Khi Huấn Luyện

Notebook hiện tại loại bỏ:

- `Posted On`
- `Point of Contact`

Lý do:

- `Posted On` chưa được biến đổi thành đặc trưng thời gian hữu ích.
- `Point of Contact` ít phản ánh trực tiếp đặc điểm vật lý hoặc vị trí của căn nhà.

### Tạo Đặc Trưng Từ Cột Floor

Cột `Floor` ban đầu được tách thành hai biến số:

- `Floor_Level`
- `Total_Floors`

Các giá trị đặc biệt như `Ground`, `Lower Basement` và `Upper Basement` được quy về `0`. Các giá trị thiếu hoặc không tách được sẽ được điền bằng median của cột tương ứng.

### Mã Hóa Area Locality

`Area Locality` có rất nhiều giá trị khác nhau, vì vậy notebook dùng frequency encoding thay vì one-hot encoding:

```python
data["Area_Locality_Freq"] = data["Area Locality"].map(locality_freq)
```

Cách này giữ lại thông tin về mức độ phổ biến của khu vực mà không tạo ra hàng nghìn cột sparse.

## Các Đặc Trưng Đang Dùng

### Numeric Features

- `BHK`
- `Size`
- `Bathroom`
- `Floor_Level`
- `Total_Floors`
- `Area_Locality_Freq`

### Ordinal Feature

- `Furnishing Status`

Thứ tự mã hóa:

```text
Unfurnished -> Semi-Furnished -> Furnished
```

### Nominal Features

- `Area Type`
- `City`
- `Tenant Preferred`

## Các Mô Hình Được So Sánh

Notebook so sánh bốn mô hình hồi quy:

1. `LinearRegression`
2. `DecisionTreeRegressor`
3. `RandomForestRegressor`
4. `GradientBoostingRegressor`

Mỗi mô hình được đặt trong pipeline tiền xử lý và tối ưu bằng:

```python
GridSearchCV(cv=5, scoring="r2", n_jobs=-1)
```

## Đánh Giá Mô Hình

Các chỉ số cuối cùng được tính trên thang giá thuê gốc:

- `MAE`
- `RMSE`
- `R2`

Notebook cũng trực quan hóa:

- Giá thuê thực tế so với giá thuê dự đoán của từng mô hình
- So sánh `Test R2`
- So sánh `Test RMSE`

Cross-validation dùng `R2` trên `log1p(Rent)` để chọn mô hình, còn metric cuối cùng được tính sau khi chuyển dự đoán về đơn vị INR.

## Tóm Tắt Quy Trình

```text
Đọc dữ liệu
-> Kiểm tra tổng quan dataset
-> Kiểm tra missing values và duplicated values
-> Phân tích phân phối Rent và outlier
-> Khám phá quan hệ giữa các đặc trưng
-> Loại bỏ một số cột
-> Tạo Floor_Level và Total_Floors
-> Frequency encoding cho Area Locality
-> Log-transform biến Rent
-> Chia train/test
-> Xây dựng pipeline tiền xử lý
-> Tối ưu mô hình bằng GridSearchCV
-> Đánh giá và trực quan hóa kết quả
-> Tổng kết mô hình tốt nhất
```

## Cấu Trúc Project

```text
.
├── House_Rent_Dataset.csv
├── House_Rent_Prediction_ML.ipynb
├── README.md
└── README_vi.md
```

## Cách Chạy

### Google Colab

Notebook hiện đang đọc dữ liệu từ Google Drive:

```python
data = pd.read_csv("/content/drive/MyDrive/ML/BTL/Datasets/House_Rent_Dataset.csv")
```

Nếu chạy trên Colab, hãy mount Google Drive và chỉnh lại đường dẫn nếu cần.

### Jupyter Trên Máy Local

Khi chạy local, thay cell đọc dữ liệu bằng:

```python
data = pd.read_csv("House_Rent_Dataset.csv")
```

Sau đó chạy notebook bằng Jupyter Notebook, JupyterLab hoặc VS Code.

## Thư Viện Sử Dụng

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
ydata-profiling
jupyter
```

Cài đặt:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling jupyter
```

## Hạn Chế Hiện Tại

- Dataset chỉ gồm sáu thành phố lớn của Ấn Độ.
- Chưa có bước lưu mô hình.
- Chưa có giao diện suy luận hoặc triển khai.
- Nếu có thêm thông tin về vị trí chi tiết, tuổi tòa nhà và tiện ích xung quanh, mô hình có thể tốt hơn.
- Kết quả phụ thuộc vào việc chạy lại notebook và chưa được xuất thành báo cáo tĩnh riêng.

## Hướng Cải Thiện

- Thử `ExtraTreesRegressor`, `HistGradientBoostingRegressor`, XGBoost, LightGBM hoặc CatBoost.
- Lưu mô hình bằng `joblib`.
- Xây dựng ứng dụng suy luận nhỏ bằng Streamlit.
- Bổ sung công cụ giải thích mô hình như SHAP hoặc permutation importance.
- Cải thiện xử lý outlier bằng robust scaling, winsorization hoặc mô hình riêng cho từng phân khúc giá.

## Tác Giả

```text
Nguyen Thi Thao My
```
