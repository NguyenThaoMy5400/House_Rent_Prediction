# House Rent Prediction

Dự án Machine Learning dự đoán giá thuê nhà và căn hộ (`Rent`) bằng các mô hình hồi quy trong `scikit-learn`.

Toàn bộ quy trình được triển khai trong [House_Rent_Prediction.ipynb](./House_Rent_Prediction.ipynb), gồm:

- Khám phá dữ liệu (EDA)
- Làm sạch dữ liệu
- Tạo đặc trưng
- Tiền xử lý bằng `Pipeline` và `ColumnTransformer`
- Tối ưu siêu tham số bằng `GridSearchCV`
- Đánh giá trên thang giá thuê gốc

## Mục tiêu dự án

Mục tiêu là dự đoán giá thuê từ các đặc trưng bất động sản có cấu trúc như:

- `BHK`
- `Size`
- `Floor`
- `Area Type`
- `City`
- `Furnishing Status`
- `Tenant Preferred`
- `Bathroom`

Phiên bản notebook hiện tại ưu tiên một pipeline đơn giản và ổn định hơn bằng cách loại bỏ một số cột được đánh giá là ít hữu ích hoặc quá tốn chi phí mã hóa.

## Dataset

Tên file dữ liệu:

```text
House_Rent_Dataset.csv
```

Nguồn dữ liệu:

Kaggle: <https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>

### Các cột gốc của bộ dữ liệu

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

## Cách mô hình hóa hiện tại

### Biến đổi biến mục tiêu

Notebook huấn luyện mô hình trên:

```python
y = np.log1p(data["Rent"])
```

Cách này giúp giảm ảnh hưởng của outlier và phân phối lệch phải. Sau khi dự đoán, kết quả được chuyển ngược bằng `np.expm1(...)` trước khi đánh giá.

### Các cột bị loại bỏ trước khi huấn luyện

Code hiện tại loại bỏ các cột sau trước khi tạo `X`:

- `Posted On`
- `Point of Contact`
- `Area Locality`

Lý do:

- `Posted On` hiện chưa được biến đổi thành đặc trưng hữu ích.
- `Point of Contact` phản ánh phía đăng tin nhiều hơn là bản thân căn nhà.
- `Area Locality` có cardinality rất cao.

### Tạo đặc trưng từ cột `Floor`

Cột `Floor` được tách thành hai biến số:

- `Floor_Level`
- `Total_Floors`

Quy ước hiện tại:

- `Ground` -> `0`
- `Upper Basement` -> `-1`
- `Lower Basement` -> `-2`

Nếu quá trình tách tạo ra giá trị không hợp lệ, notebook sẽ xóa các dòng đó bằng `dropna()`.

## Các đặc trưng đang dùng trong notebook hiện tại

### Numeric Features

- `BHK`
- `Size`
- `Bathroom`
- `Floor_Level`
- `Total_Floors`

### Categorical Features

- `Area Type`
- `City`
- `Furnishing Status`
- `Tenant Preferred`

## Các mô hình được so sánh

Notebook hiện tại so sánh 4 mô hình hồi quy:

1. `LinearRegression`
2. `DecisionTreeRegressor`
3. `RandomForestRegressor`
4. `GradientBoostingRegressor`

Tất cả các mô hình đều được đặt trong pipeline tiền xử lý và tối ưu bằng `GridSearchCV(cv=5, scoring="r2", n_jobs=-1)`.

## Đánh giá mô hình

Các chỉ số cuối cùng được tính trên thang giá thuê gốc:

- `MAE`
- `MSE`
- `RMSE`
- `R2`

Bảng kết quả trong notebook hiện báo cáo các cột:

- `Model`
- `Test MAE (Original Rent)`
- `Test MSE (Original Rent)`
- `Test RMSE (Original Rent)`
- `Test R2 (Original Rent)`

Lưu ý:

- Cross-validation vẫn dùng `R2` trên `log1p(Rent)` để chọn mô hình.

## Tóm tắt quy trình

```text
Đọc dữ liệu
-> Kiểm tra và làm sạch dữ liệu
-> Loại bỏ một số cột
-> Tạo đặc trưng từ Floor
-> Xóa các dòng lỗi khi tách Floor
-> Chia train/test
-> Xây dựng pipeline tiền xử lý
-> Tối ưu mô hình với GridSearchCV
-> Đánh giá trên thang giá thuê gốc
```

## Cấu trúc project

```text
.
├── House_Rent_Dataset.csv
├── House_Rent_Prediction.ipynb
├── README.md
└── README_vi.md
```

## Cách chạy

### Google Colab

Notebook hiện đang đọc dữ liệu từ Google Drive. Nếu chạy trên Colab, hãy kiểm tra lại đường dẫn và chạy lần lượt toàn bộ các cell.

### Jupyter trên máy local

Có thể thay cell đọc dữ liệu bằng đường dẫn local như sau:

```python
data = pd.read_csv("House_Rent_Dataset.csv")
```

Sau đó chạy notebook bằng:

- Jupyter Notebook
- JupyterLab
- VS Code

## Thư viện sử dụng

```text
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

## Hạn chế hiện tại

- `Area Locality` chưa được khai thác trong phiên bản hiện tại.
- Chưa có bước lưu mô hình.
- Chưa có giao diện suy luận hoặc triển khai.
- Kết quả phụ thuộc vào việc chạy lại notebook và chưa được xuất thành báo cáo tĩnh riêng.

## Hướng cải thiện

- Thử `ExtraTreesRegressor`, `HistGradientBoostingRegressor`, XGBoost, LightGBM hoặc CatBoost.
- Xem lại cách mã hóa `Area Locality`.
- Lưu mô hình bằng `joblib`.
- Xây dựng ứng dụng suy luận nhỏ bằng Streamlit.
- Bổ sung công cụ giải thích mô hình như SHAP.

## Tác giả

```text
Nguyen Thi Thao My
```
