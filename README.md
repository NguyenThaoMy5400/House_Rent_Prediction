# House Rent Prediction

A Machine Learning project for predicting house and apartment rental prices (`Rent`) with regression models built in `scikit-learn`.

The main workflow is implemented in [House_Rent_Prediction_ML.ipynb](./House_Rent_Prediction_ML.ipynb), including:

- Project introduction and regression problem overview
- Exploratory Data Analysis (EDA)
- Missing value and duplicate checks
- Feature engineering
- Preprocessing with `Pipeline` and `ColumnTransformer`
- Hyperparameter tuning with `GridSearchCV`
- Model evaluation on the original rent scale
- Result visualization and final discussion

## Project Objective

The goal is to predict rental prices from structured property features such as:

- `BHK`
- `Size`
- `Floor`
- `Area Type`
- `Area Locality`
- `City`
- `Furnishing Status`
- `Tenant Preferred`
- `Bathroom`

The notebook follows an end-to-end supervised regression workflow: inspect the dataset, transform raw columns into model-ready features, train multiple models, compare metrics, and select the best-performing approach.

## Dataset

Dataset file:

```text
House_Rent_Dataset.csv
```

Dataset source:

Kaggle: <https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>

### Original Columns

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

## Current Modeling Approach

### Target Transformation

The notebook trains models on:

```python
y = np.log1p(data["Rent"])
```

This reduces the impact of high-rent outliers and the strong right skew in `Rent`. Predictions are converted back to the original rent scale with `np.expm1(...)` before final evaluation.

### Columns Removed Before Training

The current notebook removes:

- `Posted On`
- `Point of Contact`

Rationale:

- `Posted On` is not currently engineered into time-based features.
- `Point of Contact` is less directly related to the physical and location characteristics of a rental property.

### Floor Feature Engineering

The raw `Floor` column is parsed into two numeric features:

- `Floor_Level`
- `Total_Floors`

Special floor values such as `Ground`, `Lower Basement`, and `Upper Basement` are mapped to `0`. Missing or invalid parsed values are filled with the median of the corresponding engineered column.

### Area Locality Encoding

`Area Locality` has high cardinality, so the notebook uses frequency encoding instead of one-hot encoding:

```python
data["Area_Locality_Freq"] = data["Area Locality"].map(locality_freq)
```

This keeps locality information while avoiding thousands of sparse dummy columns.

## Features Used In The Current Notebook

### Numeric Features

- `BHK`
- `Size`
- `Bathroom`
- `Floor_Level`
- `Total_Floors`
- `Area_Locality_Freq`

### Ordinal Feature

- `Furnishing Status`

Encoding order:

```text
Unfurnished -> Semi-Furnished -> Furnished
```

### Nominal Features

- `Area Type`
- `City`
- `Tenant Preferred`

## Models Evaluated

The notebook compares four regression models:

1. `LinearRegression`
2. `DecisionTreeRegressor`
3. `RandomForestRegressor`
4. `GradientBoostingRegressor`

Each model is wrapped in a preprocessing pipeline and tuned with:

```python
GridSearchCV(cv=5, scoring="r2", n_jobs=-1)
```

## Evaluation

Final metrics are computed on the original rent scale:

- `MAE`
- `RMSE`
- `R2`

The notebook also visualizes:

- Actual vs predicted rent for each model
- Test `R2` comparison
- Test `RMSE` comparison

Cross-validation uses `R2` on `log1p(Rent)` internally for model selection, while final test metrics are reported after converting predictions back to INR.

## Workflow Summary

```text
Load data
-> Inspect dataset
-> Check missing values and duplicates
-> Analyze Rent distribution and outliers
-> Explore feature relationships
-> Drop selected columns
-> Engineer Floor_Level and Total_Floors
-> Frequency-encode Area Locality
-> Apply log-transform to Rent
-> Split train/test
-> Build preprocessing pipeline
-> Tune models with GridSearchCV
-> Evaluate and visualize results
-> Summarize the best model
```

## Project Structure

```text
.
├── House_Rent_Dataset.csv
├── House_Rent_Prediction_ML.ipynb
├── README.md
└── README_vi.md
```

## How To Run

### Google Colab

The notebook currently reads data from Google Drive:

```python
data = pd.read_csv("/content/drive/MyDrive/ML/BTL/Datasets/House_Rent_Dataset.csv")
```

If you use Colab, mount Google Drive and update the path if needed.

### Local Jupyter Environment

For local execution, replace the data-loading cell with:

```python
data = pd.read_csv("House_Rent_Dataset.csv")
```

Then run the notebook in Jupyter Notebook, JupyterLab, or VS Code.

## Libraries Used

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
ydata-profiling
jupyter
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling jupyter
```

## Current Limitations

- The dataset covers only six major Indian cities.
- No model persistence is implemented.
- No deployment or prediction interface is included.
- Additional location, building age, and amenity features could improve real-world performance.
- Results depend on rerunning the notebook and are not exported to a fixed report file.

## Possible Improvements

- Try `ExtraTreesRegressor`, `HistGradientBoostingRegressor`, XGBoost, LightGBM, or CatBoost.
- Add model saving with `joblib`.
- Build a small inference app with Streamlit.
- Add model interpretation tools such as SHAP or permutation importance.
- Improve outlier handling with robust scaling, winsorization, or segment-specific models.

## Author

```text
Nguyen Thi Thao My
```
