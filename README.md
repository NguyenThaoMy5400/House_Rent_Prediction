# House Rent Prediction

A Machine Learning project for predicting house and apartment rental prices (`Rent`) with regression models built in `scikit-learn`.

The full workflow is implemented in [House_Rent_Prediction.ipynb](./House_Rent_Prediction.ipynb), including:

- Exploratory Data Analysis (EDA)
- Data cleaning
- Feature engineering
- Preprocessing with `Pipeline` and `ColumnTransformer`
- Hyperparameter tuning with `GridSearchCV`
- Evaluation on the original rent scale

## Project Objective

The goal is to predict rental prices from structured property features such as:

- `BHK`
- `Size`
- `Floor`
- `Area Type`
- `City`
- `Furnishing Status`
- `Tenant Preferred`
- `Bathroom`

The current notebook version focuses on a simpler and more stable pipeline by dropping some columns that were judged less useful or too costly to encode.

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

This helps reduce the effect of outliers and right skew. Predictions are converted back with `np.expm1(...)` before final evaluation.

### Columns Removed Before Training

The current code drops these columns before building `X`:

- `Posted On`
- `Point of Contact`
- `Area Locality`

Rationale:

- `Posted On` is not currently engineered into useful features.
- `Point of Contact` mainly reflects the listing side rather than the property itself.
- `Area Locality` has very high cardinality and is intentionally excluded to keep the pipeline simpler.

### Floor Feature Engineering

The `Floor` column is transformed into two numeric features:

- `Floor_Level`
- `Total_Floors`

Special values are mapped as follows:

- `Ground` -> `0`
- `Upper Basement` -> `-1`
- `Lower Basement` -> `-2`

Rows with invalid values created during this parsing step are removed with `dropna()`.

## Features Used In The Current Notebook

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

## Models Evaluated

The notebook compares four regression models:

1. `LinearRegression`
2. `DecisionTreeRegressor`
3. `RandomForestRegressor`
4. `GradientBoostingRegressor`

All models are wrapped in a preprocessing pipeline and tuned with `GridSearchCV(cv=5, scoring="r2", n_jobs=-1)`.

## Evaluation

Final metrics are computed on the original rent scale:

- `MAE`
- `MSE`
- `RMSE`
- `R2`

The results table in the notebook currently reports:

- `Model`
- `Test MAE (Original Rent)`
- `Test MSE (Original Rent)`
- `Test RMSE (Original Rent)`
- `Test R2 (Original Rent)`

Note:

- Cross-validation still uses `R2` on `log1p(Rent)` internally for model selection.
- The notebook no longer reports `Best CV R2 (log Rent)` in the final performance summary table.

## Workflow Summary

```text
Load data
-> Inspect and clean data
-> Drop selected columns
-> Engineer floor features
-> Remove rows with invalid parsed floor values
-> Split train/test
-> Build preprocessing pipeline
-> Tune models with GridSearchCV
-> Evaluate on original rent scale
```

## Project Structure

```text
.
├── House_Rent_Dataset.csv
├── House_Rent_Prediction.ipynb
├── README.md
└── README_vi.md
```

## How To Run

### Google Colab

The notebook currently reads data from Google Drive. If you use Colab, update the path if needed and run all cells from top to bottom.

### Local Jupyter Environment

Replace the data-loading cell with a local path such as:

```python
data = pd.read_csv("House_Rent_Dataset.csv")
```

Then run the notebook in:

- Jupyter Notebook
- JupyterLab
- VS Code

## Libraries Used

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Current Limitations

- `Area Locality` is not modeled in the current version.
- No model persistence is implemented.
- No deployment or prediction interface is included.
- Results depend on rerunning the notebook and are not exported to a fixed report file.

## Possible Improvements

- Try `ExtraTreesRegressor`, `HistGradientBoostingRegressor`, XGBoost, LightGBM, or CatBoost.
- Revisit `Area Locality` with a better encoding strategy.
- Add model saving with `joblib`.
- Build a small inference app with Streamlit.
- Add model interpretation tools such as SHAP.

## Author

```text
Nguyen Thi Thao My
```
