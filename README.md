# 🏠 House Rent Prediction

A Machine Learning project for predicting **house/apartment rental prices (Rent)** using various regression models implemented with **scikit-learn**.

The complete workflow, including Exploratory Data Analysis (EDA), data preprocessing, feature engineering, categorical encoding, pipeline construction, hyperparameter tuning using **GridSearchCV**, and model evaluation, is implemented in the notebook:

```text
House_Rent_Prediction_final.ipynb
```

---

# 📌 Project Objective

The objective of this project is to build a regression model capable of predicting rental prices (`Rent`) based on property-related features such as:

* Number of bedrooms (`BHK`)
* Property size (`Size`)
* Current floor and total floors (`Floor`)
* Area type (`Area Type`)
* Detailed location (`Area Locality`)
* City (`City`)
* Furnishing status (`Furnishing Status`)
* Preferred tenant category (`Tenant Preferred`)
* Number of bathrooms (`Bathroom`)
* Point of contact (`Point of Contact`)
* Posting date (`Posted On`)

In addition to building a predictive model, this project addresses common real-world data challenges such as:

* Right-skewed target distribution (`Rent`)
* High-cardinality categorical features (`Area Locality`)
* Preventing data leakage during preprocessing

---

# 📂 Dataset

**Dataset File**

```text
House_Rent_Dataset.csv
```

**Dataset Source**

Kaggle: <https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>

## Overview

| Attribute                      | Value |
| ------------------------------ | ----- |
| Number of Rows                 | 4746  |
| Number of Columns              | 12    |
| Target Variable                | Rent  |
| Unique Values in Area Locality | 2235  |

Some cities included in the dataset:

* Mumbai
* Chennai
* Bangalore
* Hyderabad
* Delhi
* Kolkata

## Dataset Features

| Feature           | Description           |
| ----------------- | --------------------- |
| Posted On         | Property listing date |
| BHK               | Number of bedrooms    |
| Rent              | Rental price          |
| Size              | Property size         |
| Floor             | Floor information     |
| Area Type         | Area category         |
| Area Locality     | Detailed location     |
| City              | City                  |
| Furnishing Status | Furnishing condition  |
| Tenant Preferred  | Preferred tenant type |
| Bathroom          | Number of bathrooms   |
| Point of Contact  | Contact person type   |

---

# ⚙️ Machine Learning Pipeline

The project follows a complete Machine Learning workflow:

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

## Main Steps

### 1. Data Loading

Load the dataset from a CSV file.

### 2. Exploratory Data Analysis (EDA)

* Check dataset shape
* Inspect data types
* Check missing values
* Check duplicate records
* Analyze the distribution of `Rent`
* Visualize relationships between features and rental prices

### 3. Feature Engineering

#### Processing `Posted On`

Convert to datetime format and extract:

* Posted_Month
* Posted_Day
* Posted_DayOfWeek

#### Processing `Floor`

Split into:

* Floor_Level
* Total_Floors

Example:

```text
3 out of 5
```

becomes:

```text
Floor_Level = 3
Total_Floors = 5
```

---

# 🔧 Advanced Feature Engineering

## 1. Log Transformation of Rent

The target variable `Rent` is heavily right-skewed and contains many outliers.

To reduce the impact of extreme values, the model is trained on:

```python
y = np.log1p(data["Rent"])
```

After prediction:

```python
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)
```

### Benefits

* Reduces the influence of outliers
* Improves model stability
* Facilitates optimization for rental price prediction

---

## 2. Frequency Encoding for Area Locality

The feature:

```text
Area Locality
```

contains:

```text
2235 unique values
```

Applying One-Hot Encoding directly would create an extremely high-dimensional feature space.

Instead, Frequency Encoding is used:

```python
area_locality_freq_map = X_train["Area Locality"].value_counts(normalize=True)
```

A new feature is created:

```python
Area_Locality_Freq
```

### Advantages

* Prevents dimensionality explosion
* Reduces overfitting risk
* Saves memory
* Retains location-related information

### Preventing Data Leakage

The frequency map is computed using only the training set:

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

# 📊 Features Used

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

# 🤖 Models Evaluated

The project compares four regression algorithms:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

All models are optimized using:

```python
GridSearchCV(
    cv=5,
    scoring="r2",
    n_jobs=-1
)
```

---

# 📈 Model Evaluation

Evaluation metrics are calculated on the original rental prices after reversing the log transformation:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

## Metric Descriptions

| Metric | Description                               |
| ------ | ----------------------------------------- |
| MAE    | Average absolute prediction error         |
| MSE    | Average squared prediction error          |
| RMSE   | Prediction error in the same unit as Rent |
| R²     | Variance explained by the model           |

---

# 🏆 Experimental Results

| Model                       | Best CV R² (Log Rent) |  Test MAE | Test RMSE | Test R² |
| --------------------------- | --------------------: | --------: | --------: | ------: |
| Linear Regression           |                0.8120 | 11,396.86 | 31,826.16 |  0.7533 |
| Gradient Boosting Regressor |                0.8322 | 11,551.20 | 37,284.89 |  0.6614 |
| Random Forest Regressor     |                0.8254 | 11,364.49 | 40,178.34 |  0.6068 |
| Decision Tree Regressor     |                0.7763 | 12,808.13 | 40,675.65 |  0.5970 |

---

# 🥇 Best Model

Based on test-set performance, the best model is:

## Linear Regression

### Best Parameters

```python
{'regressor__fit_intercept': True}
```

### Final Results

| Metric |            Value |
| ------ | ---------------: |
| MAE    |        11,396.86 |
| MSE    | 1,012,904,643.11 |
| RMSE   |        31,826.16 |
| R²     |           0.7533 |

Although Gradient Boosting achieved a higher cross-validation score on the log-transformed target, Linear Regression produced the best performance when evaluated on the original rental prices.

---

# 📁 Project Structure

```text
.
├── House_Rent_Prediction_final.ipynb
├── House_Rent_Dataset.csv
├── README.md
├── README_EN.md
└── model_comparison_results.csv
```

---

# 🚀 How to Run

## Google Colab

```python
from google.colab import drive

drive.mount('/content/drive')

data = pd.read_csv(
    "/content/drive/MyDrive/ML/BTL/Datasets/House_Rent_Dataset.csv"
)
```

### Steps

1. Upload the notebook to Google Colab.
2. Place the CSV file in the correct Google Drive location.
3. Run all notebook cells from top to bottom.

---

## Local Environment

Modify the data loading cell:

```python
import pandas as pd

data = pd.read_csv("House_Rent_Dataset.csv")
```

Then run the notebook using:

* Jupyter Notebook
* VS Code
* Jupyter Lab

---

# 📦 Libraries Used

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Installation:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

# ✅ Project Strengths

* Complete Machine Learning pipeline
* Effective feature engineering
* Log transformation for target variable
* Frequency encoding for high-cardinality features
* Hyperparameter tuning with GridSearchCV
* Comparison of multiple regression algorithms
* Evaluation on original rental prices

---

# ⚠️ Current Limitations

* Advanced boosting models have not been explored
* No real-time prediction system
* Model persistence has not been implemented

---

# 🔮 Future Improvements

Potential future enhancements include:

* Adding XGBoost
* Adding LightGBM
* Adding CatBoost
* Saving trained models with Joblib
* Building a Streamlit web application
* Using SHAP for model interpretability
* Further outlier optimization

---

# 👨‍💻 Author

```text
Author : Nguyen Thi Thao My
Project: House Rent Prediction
```

---

# 📜 License

This project may be distributed under:

```text
MIT License
```
