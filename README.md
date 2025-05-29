# Telco Customer Churn Prediction

This project aims to predict customer churn in a telecommunications company using various machine learning models. Understanding the factors that lead to customer churn can help businesses develop targeted strategies to retain customers and improve their services.

## Project Objectives

* **Analyze** the provided Telco customer dataset to identify patterns and insights related to customer churn.
* **Develop** machine learning models to accurately predict whether a customer will churn.
* **Evaluate** the performance of different models and identify the most effective one for churn prediction.
* **Extract** key insights regarding feature importance to understand the drivers of churn.

## Project Structure

├── telco_churn_analysis.py
├── Telco customer churn.csv
├── README.md


* `telco_churn_analysis.py`: Contains the Python code for data loading, understanding, EDA, preprocessing, model training, and evaluation.
* `Telco customer churn.csv`: The dataset used for this project.
* `README.md`: This file, providing an overview of the project.

## Exploratory Data Analysis (EDA)

The EDA phase involved visualizing key aspects of the data to understand the churn distribution and its relationship with various features.

### Key Findings:

* **Churn Distribution**: The dataset shows an imbalanced churn distribution, with a higher number of customers not churning compared to those who do. This imbalance was addressed during preprocessing using SMOTE.
    ![Churn Distribution](https://i.imgur.com/example_churn_distribution.png) * **Churn by Senior Citizen**: Senior citizens appear to have a slightly higher churn rate compared to non-senior citizens.
    ![Churn by Senior Citizen](https://i.imgur.com/example_seniorcitizen_churn.png) * **Churn by Contract Type**: Customers with month-to-month contracts exhibit a significantly higher churn rate compared to those with one-year or two-year contracts, suggesting contract duration is a strong predictor of churn.
    ![Churn by Contract Type](https://i.imgur.com/example_contract_churn.png) ## Preprocessing Pipeline

The preprocessing steps were crucial for preparing the raw data for model training.

* **Missing Values Handling**: `TotalCharges` column, initially an object type, was converted to numeric, and missing values (introduced by coercion errors) were filled with 0.
* **Feature Dropping**: The `customerID` column was removed as it serves no analytical purpose.
* **Target Encoding**: The `Churn` column was converted from categorical ('Yes', 'No') to numerical (1, 0).
* **Categorical Encoding**: All other categorical features were one-hot encoded using `pd.get_dummies` to transform them into a numerical format suitable for machine learning models.
* **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets.
* **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to address the class imbalance in the `Churn` target variable, generating synthetic samples for the minority class.

## Model Training & Performance

Three different machine learning models were trained and evaluated for churn prediction: Logistic Regression, Random Forest, and XGBoost. SMOTE was applied to the training data to mitigate the class imbalance issue.

| Model Name        | AUC Score (Test Set) |
| :---------------- | :------------------- |
| Logistic Regression | ~0.84                |
| Random Forest     | ~0.82                |
| XGBoost           | ~0.83                |

*Note: The exact AUC scores may vary slightly based on the random state and data split.*

## Evaluation Metrics

The models were evaluated primarily using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) score, and a classification report which includes precision, recall, and F1-score for each class.

### Example Classification Report (for a representative model):

          precision    recall  f1-score   support

       0       0.90      0.88      0.89      1036
       1       0.60      0.65      0.62       373

accuracy                           0.82      1409
macro avg       0.75      0.76      0.76      1409
weighted avg       0.82      0.82      0.82      1409


### Example Confusion Matrix (for a representative model):

Predicted 0   Predicted 1
True 0     [[TN, FP],
True 1      [FN, TP]]


## Hyperparameter Tuning

While the provided code doesn't explicitly include hyperparameter tuning (e.g., GridSearchCV/RandomizedSearchCV), in a production setting, this step would be crucial to optimize model performance. For instance, for a Random Forest Classifier, one might tune parameters like `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

```python
# Example of GridSearchCV for RandomForestClassifier (not in current script)
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid_search.fit(X_train_res, y_train_res)
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best AUC score: {grid_search.best_score_}")
Model Export
For deployment, the trained model (e.g., the best-performing one) would typically be saved using pickle or joblib.

Python
# Example of model export (not in current script)
# import joblib
# joblib.dump(best_model, 'telco_churn_prediction_model.pkl')
Key Insights & Summary
Contract Type is Key: Month-to-month contracts are highly associated with churn. This suggests that offering incentives for longer-term contracts could reduce churn.
Senior Citizens & Churn: Senior citizens show a slightly higher propensity to churn. Tailored retention strategies might be beneficial for this demographic.
Importance of TotalCharges and MonthlyCharges: These features generally appear as top predictors across models, indicating that pricing and billing aspects significantly influence churn.
Internet Service Impact: Features related to internet service (e.g., OnlineSecurity, TechSupport) are also important, highlighting the need for reliable and secure internet services.
Addressing Imbalance: SMOTE effectively helped in building more robust models by balancing the target classes.
Final Reflections
This project successfully built and evaluated several machine learning models for Telco customer churn prediction. The insights gained from feature importance can guide business decisions, focusing on customer retention strategies for high-risk segments. Future work could involve more extensive feature engineering, advanced hyperparameter tuning, and exploring deep learning models for potentially higher accuracy.

How to Use This Repo
Clone the repository:
Bash
git clone [https://github.com/your-username/telco-churn-prediction.git](https://github.com/your-username/telco-churn-prediction.git)
cd telco-churn-prediction
Install dependencies:
Bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
Run the analysis script:
Bash
python telco_churn_analysis.py
This script will perform data loading, understanding, EDA, preprocessing, model training, and print the evaluation results for each model.
References
Telco Customer Churn Dataset: Kaggle
Scikit-learn Documentation: https://scikit-learn.org/
Imbalanced-learn Documentation: https://imbalanced-learn.org/
XGBoost Documentation: https://xgboost.readthedocs.io/
<!-- end list -->
