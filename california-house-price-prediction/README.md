# California Housing Price Prediction

A machine learning project to predict house values in California using regression models and feature importance analysis.

## Project Overview

This project builds a predictive model to estimate housing prices in California based on data features. By understanding the key drivers of housing values and generating accurate predictions, real estate companies, investors, or policymakers can make informed decisions about property valuation, market trends, and urban planning.

## Problem Statement

Accurate housing price prediction is essential for real estate markets, but factors like location, demographics and property characteristics create complex relationships. The dataset shows high variability in prices, with capping at $500,000 affecting ~4.5% of samples. This project aims to:

1. **Identify key factors** driving housing prices
2. **Build a predictive model** to estimate house values
3. **Provide insights** for strategic decision-making

## Dataset

**Source**: California Housing Dataset (20,640 records)

**Features**: 10 attributes (9 features + 1 target variable) covering:
- Geographic location (longitude, latitude)
- Property characteristics (housing median age, total rooms, total bedrooms)
- Demographics (population, households)
- Economic factors (median income)
- Proximity to ocean (categorical: INLAND, NEAR BAY, etc.)

**Target Variable**: Median house value (capped at $500,001; mean ~$206,856)

## Project Structure

```
california-housing-price-prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                          # Original dataset
│   │   └── california-housing.csv
│   └── processed/                    # Preprocessed data splits
│       ├── X_train.csv               # Training features
│       ├── X_val.csv                 # Validation features
│       ├── X_test.csv                # Test features
│       ├── y_train.csv               # Training target
│       ├── y_val.csv                 # Validation target
│       └── y_test.csv                # Test target
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb    # Data exploration & cleaning
│   └── 02_model_training_and_evaluation.ipynb  # Model development & evaluation
└── models/                           # Trained models
    ├── final_california_price_prediction_model_xgboost.pkl
```

## Methodology

### 1. Exploratory Data Analysis (Notebook 01)

- **Data Quality**: Checked for missing values (~1% in total_bedrooms), duplicates, and inconsistent types
- **Feature Analysis**: 
  - Identified 9 numerical and 1 categorical feature
  - Analyzed distributions, correlations, and relationships with price
  - Detected multicollinearity (e.g., total_rooms and households) and skewness
- **Key Insights**:
  - Strong geographic clustering (prices higher near coast)
  - Positive correlation with median income
  - Right-skewed distributions in rooms, population, and prices
  - ~4.5% of prices capped at $500,001, potentially biasing high-end predictions

### 2. Data Preprocessing

- **Data Cleaning**:
  - Imputed missing total_bedrooms with median
  - Removed ~10.8% multivariate outliers using Isolation Forest
  - One-hot encoded ocean_proximity

- **Feature Engineering**:
  - Created 3 ratio features: rooms_per_household, bedrooms_per_room, population_per_household
  - Standardized all numerical features using StandardScaler (fit on training data only)

- **Data Splitting**:
  - Stratified split: 60% train, 20% validation, 20% test (preserving price distribution)
  - Ensured no data leakage between splits

### 3. Model Development (Notebook 02)

- **Baseline Models** (5-fold stratified cross-validation):
  - Random Forest
  - XGBoost
  - Gradient Boosting

- **Hyperparameter Tuning**:
  - GridSearchCV on two best performing models
  - Evaluation metric: R² (explains variance) and RMSE (error magnitude)
  - Tested multiple hyperparameter combinations for both models

- **Final Evaluation**:
  - Test set performance using multiple metrics
  - Residuals analysis
  - Actual vs predicted price analysis
  - Feature importance analysis

## Results

### Best Model: XGBoost

| Metric | Score |
|--------|-------|
| **R²** | 0.8412 |
| **RMSE** | $44,592 |
| **MAE** | $28,165 |
| **MAPE** | 0.1540 |

**Interpretation**:
- Explains ~84% of price variance (R²: 0.84)
- Average absolute error of ~$28,165 (MAE)
- Relative error of ~15% (MAPE)
- Performs best in mid-range properties (100.000$ – 400.000$); slight over-prediction for low-end and under-prediction for high-end properties (mostly due to capping)

### Top Price Drivers

1. **Location**: Ocean proximity (INLAND: ~56.9% importance) - big changes in pricing based on location
2. **Income**: Median income (~11.8% importance) - Strong positive correlation with values
3. **Engineered Ratios**: Rooms per household and bedrooms per room add meaningful value
4. **Age and Size**: Housing median age and total rooms contribute moderately
5. **Demographics**: Population, households and room counts have minimal impact (<1% each)

The **top 8 features** account for ~90% of model decisions, indicating focused geographic and economic drivers.

## Insights and Recommendations

1. **Investment Targeting**
   - Prioritize coastal and near-ocean properties for higher valuations
   - Focus on high-income areas for premium properties

2. **Market Analysis**
   - Monitor inland vs. coastal price gaps for opportunity spotting
   - Target mid-range properties where model is most accurate

3. **Risk Assessment**
   - Careful planning around properties with high prediction errors (e.g., capped high-end)
   - Update model periodically with fresh data to handle market changes

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **scikit-learn**: Machine learning models, metrics, and preprocessing
- **xgboost**: Gradient boosting regressor
- **scipy**: Statistical functions
- **joblib**: Model serialization

See `requirements.txt` for full dependency list.

## Project Information

**Objective**: Predict housing prices and identify key market drivers

**Dataset**: 20,640 California housing records

**Best Model**: XGBoost (R²: 0.8412)

**Output**: Price predictions and feature importance for real estate strategy