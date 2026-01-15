# Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecommunications company using classification models and feature importance analysis.

## Project Overview

This project builds a predictive model to identify customers at high risk of churning (leaving the company). By understanding the key drivers of churn and predicting at-risk customers, the telecommunications company can implement targeted retention strategies and improve customer lifetime value.

## Problem Statement

Customer churn is a critical business challenge for telecommunications companies. With a baseline churn rate of 26.5%, the company loses significant revenue and market share. This project aims to:

1. **Identify key factors** driving customer churn
2. **Build a predictive model** to flag at-risk customers
3. **Provide business insights** for strategic decision-making

## Dataset

**Source**: Telco Customer Churn Dataset (7,043 customer records)

**Features**: 21 attributes covering:
- Customer demographics (age, relationship status, dependents)
- Account information (tenure, contract type)
- Service usage (internet service, add-ons like security, backup, tech support)
- Billing (monthly charges, total charges, payment method)

**Target Variable**: Churn (26.5% churn rate - moderately imbalanced)

## Project Structure

```
telco-customer-churn-prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                          # Original dataset
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/                    # Preprocessed data splits
        ├── processed_data.csv        # Processed dataset
│       ├── X_train_balanced.csv      # Training features (SMOTE-balanced)
│       ├── X_val.csv                 # Validation features
│       ├── X_test.csv                # Test features
│       ├── y_train_balanced.csv      # Training target (SMOTE-balanced)
│       ├── y_val.csv                 # Validation target
│       └── y_test.csv                # Test target
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb    # Data exploration & cleaning
│   └── 02_model_training_and_evaluation.ipynb  # Model development & evaluation
└── models/                           # Trained models
    ├── final_churn_model_randomforest.pkl
```

## Methodology

### 1. Exploratory Data Analysis (Notebook 01)

- **Data Quality**: Checked for missing values, duplicates, and inconsistent types
- **Feature Analysis**: 
  - Identified 17 categorical and 3 numerical features
  - Analyzed distributions and relationships with churn
  - Dropped 5 low-relevance features (gender, phone service, streaming services)
- **Key Insights**:
  - Tenure shows bimodal distribution (new vs. established customers)
  - Fiber optic internet and electronic check payments correlate with high churn
  - Protective services (security, backup, tech support) reduce churn risk
  - Longer contracts significantly improve retention

### 2. Data Preprocessing

- **Data Cleaning**:
  - Fixed TotalCharges data type (text → numeric)
  - Imputed missing TotalCharges with 0 (new customers not yet billed)
  - Standardized categorical encoding (SeniorCitizen: numeric → text)

- **Feature Engineering**:
  - One-hot encoded 11 relevant categorical features
  - Standardized numerical features (tenure, monthly charges)
  - Removed 5 highly collinear features to reduce multicollinearity

- **Class Imbalance Handling**:
  - Applied SMOTE oversampling to training set only
  - Balanced training data from 26.5% to 50% churn rate
  - Preserved authentic validation/test distributions

- **Data Splitting**:
  - Stratified split: 70% train, 15% validation, 15% test
  - Ensured no data leakage between splits

### 3. Model Development (Notebook 02)

- **Baseline Models** (5-fold stratified cross-validation):
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)

- **Hyperparameter Tuning**:
  - GridSearchCV on Random Forest and XGBoost (best performers)
  - Evaluation metric: ROC-AUC (robust to class imbalance)
  - Tested 60+ hyperparameter combinations for each model

- **Final Evaluation**:
  - Test set performance using multiple metrics
  - Confusion matrix analysis
  - Feature importance rankings
  - Precision-Recall curves

## Results

### Best Model: Random Forest

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.8205 |
| **Accuracy** | High |
| **Precision** | 57% |
| **Recall** | 65% |
| **F1-Score** | 0.6057 |

**Interpretation**:
- Excellent discrimination between churners and non-churners (ROC-AUC: 0.82)
- Catches ~65% of actual churners (recall)
- ~57% of flagged customers actually churn (precision)
- Solid balance between precision and recall (F1: 0.61)

### Top Churn Drivers

1. **Tenure**: Newer customers (first 6-12 months) at highest risk
2. **Monthly Charges**: Higher charges correlate with churn
3. **Contract Type**: Month-to-month contracts have 3-5x higher churn
4. **Internet Service**: Fiber optic shows high churn correlation
5. **Protective Services**: Security/backup reduce churn risk

The **top 12 features** account for ~90% of model decisions, indicating a focused and interpretable set of churn drivers.

## Business Recommendations

1. **Early Intervention Program**
   - Target customers in first 6-12 months with engagement campaigns
   - Offer onboarding incentives and support

2. **Pricing Strategy**
   - Analyze fiber optic pricing; consider competitive adjustments
   - Offer personalized discounts for high-charge segments

3. **Contract Incentives**
   - Promote long-term contracts with discounts (1-2 year locks)
   - Make month-to-month less attractive

4. **Service Add-on Push**
   - Bundle security, backup, and tech support services
   - Emphasize churn-reducing benefits to new customers

5. **Proactive Outreach**
   - Score customers monthly using the churn model
   - Target top-risk segments with personalized retention offers

6. **Payment Method Optimization**
   - Encourage automatic payments (reduce electronic check usage)
   - Offer incentives for bank transfers or credit cards


## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models and metrics
- **xgboost**: Gradient boosting classifier
- **imbalanced-learn**: SMOTE for handling class imbalance
- **matplotlib/seaborn**: Data visualization
- **joblib**: Model serialization

See `requirements.txt` for full dependency list.

## Project Information

**Objective**: Predict customer churn and identify key business drivers

**Dataset**: 7,043 telecommunications customer records

**Best Model**: Random Forest (ROC-AUC: 0.8205)

**Output**: Churn predictions and feature importance for business strategy