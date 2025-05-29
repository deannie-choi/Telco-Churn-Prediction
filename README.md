# Telco Customer Churn Prediction

## Project Objectives
This project aims to predict customer churn in a telecommunications company using machine learning models. The objective is to identify at-risk customers so that the company can take proactive steps to retain them.

## Project Structure
- `telco_churn_analysis.py`: Main script containing all steps from data loading to model evaluation
- `Telco customer churn.csv`: Dataset used for training and evaluation

## Exploratory Data Analysis (EDA)
- **Churn Distribution**: Significant class imbalance observed
- **Key Features**:
  - Higher churn among senior citizens
  - Customers with month-to-month contracts showed higher churn
- **Visualizations**: Bar plots for churn rate by customer attributes

## Preprocessing Pipeline
- Convert `TotalCharges` to numeric and fill missing values
- Drop `customerID` column
- Encode `Churn` column into binary values (0: No, 1: Yes)
- One-hot encode categorical variables
- Train-test split (80/20)
- Apply SMOTE for class balancing on training set

## Model Training & Performance
Three models were trained and compared:

| Model              | ROC-AUC Score |
|-------------------|---------------|
| Logistic Regression | ~0.84         |
| Random Forest      | ~0.87         |
| XGBoost            | ~0.89         |

## Evaluation Metrics
- Metrics used: ROC-AUC, confusion matrix, classification report
- Example insights:
  - XGBoost had the highest overall precision and recall
  - Logistic Regression performed well but slightly under Random Forest

## Hyperparameter Tuning
While this script used default parameters, a future step would include `GridSearchCV` or `RandomizedSearchCV` for optimal hyperparameters, especially for Random Forest and XGBoost.

## Model Export
Model export functionality is not yet implemented. Suggested next steps include using `joblib` or `pickle` to serialize the best-performing model.

## Key Insights & Summary
- Contract type, tenure, and total charges are strong indicators of churn
- SMOTE improved model fairness by balancing the dataset
- XGBoost consistently outperformed other models in terms of accuracy and ROC-AUC

## Final Reflections
This project provides a strong baseline for churn prediction. Further improvements could include feature importance analysis, SHAP values for explainability, and advanced ensemble techniques.

## How to Use This Repo
1. Place `Telco customer churn.csv` in the working directory
2. Run `telco_churn_analysis.py`
3. Review printed metrics and plots

## References
- Kaggle Telco Customer Churn Dataset
- Scikit-learn Documentation
- XGBoost Documentation
- imbalanced-learn SMOTE
```

