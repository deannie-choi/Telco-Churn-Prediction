# Telco Customer Churn Prediction 📞

This project is a complete end-to-end machine learning case study using the Telco Customer Churn dataset from Kaggle. It includes thorough data cleaning, exploration, preprocessing, modeling, evaluation, and export. The goal is to predict whether a telecom customer will churn based on their demographics, service usage, and billing information.

---

## 🎯 **Project Objectives**

- Identify the key factors that contribute to customer churn in telecom.
- Explore churn trends using visual analysis.
- Clean and preprocess the data using ML-ready practices.
- Train and compare multiple classification models.
- Evaluate model performance and save the best one for reuse.

---

## 📊 **Exploratory Data Analysis (EDA)**

Explored churn distribution and how it's affected by key features.

### 🧭 Key Findings

- Customers with month-to-month contracts churned more frequently.
- Higher monthly charges correlated with higher churn.
- Tenure was significantly shorter among churned customers.
- Senior citizens and people without partners or dependents had higher churn rates.

---

## 🛠️ **Preprocessing Pipeline**

- Converted `TotalCharges` from `object` to numeric and filled missing values with 0.
- Converted `SeniorCitizen` to categorical.
- Dropped non-informative features like `customerID`.
- Applied one-hot encoding to categorical variables.
- Scaled numerical variables using `StandardScaler`.

---

## 🤖 **Model Training & Performance**

### 🧪 Models Trained:
- Logistic Regression (baseline)
- XGBoost Classifier

### 📈 Accuracy Comparison:

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 79.1%    |
| XGBoost Classifier  | 82.4% ✅ |

---

## 🧪 **Evaluation Metrics**

Confusion matrix and classification report used to evaluate XGBoost:

```python
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
```

```
              precision    recall  f1-score   support
           0       0.85      0.89      0.87       999
           1       0.72      0.63      0.67       347

    accuracy                           0.82      1346
   macro avg       0.79      0.76      0.77      1346
weighted avg       0.81      0.82      0.81      1346
```

---

## 🔧 **Hyperparameter Tuning**

Used `RandomizedSearchCV` for XGBoost tuning:

```python
param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [3, 5, 7],
  'learning_rate': [0.01, 0.1],
  'subsample': [0.8, 1.0]
}
```

✅ Best model achieved **82.4% accuracy** with improved recall for churn class.

---

## 📌 **Key Insights & Summary**

| Insight             | Conclusion                                       |
|---------------------|--------------------------------------------------|
| Contract Type       | Month-to-month customers churn more              |
| Monthly Charges     | High charges increase likelihood of churn        |
| Tenure              | Short-term customers are more likely to churn    |
| Senior Citizen      | Slightly higher churn rate                       |
| Living Alone        | Customers without partners/dependents churn more |

---

## 🧠 **Final Reflections**

This project demonstrates:
- The value of exploratory analysis in uncovering churn patterns  
- The importance of feature engineering (e.g., tenure, contract type)  
- Reliable model building with clear evaluation strategies  
- A professional machine learning workflow applicable to real-world cases

---

⭐️ If you found this project helpful, feel free to star the repo. Thank you!

