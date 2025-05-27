import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load data
def load_data():
    df = pd.read_csv('Telco customer churn.csv')
    return df

# Data understanding
def data_understanding(df):
    print("\nData Basic Information")
    print(df.info())
    print("\nData Statistics")
    print(df.describe())
    print("\nChurn Rate")
    print(df['Churn'].value_counts(normalize=True))

# Exploratory Data Analysis (EDA)
def exploratory_data_analysis(df):
    # Churn Rate Visualization
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.show()

    # Churn Rate by Age Group
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='SeniorCitizen', hue='Churn')
    plt.title('Churn by SeniorCitizen')
    plt.show()

    # 
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Churn by Contract Type')
    plt.xticks(rotation=45)
    plt.show()

# Data preprocessing
def preprocessing(df):
    # Handle missing values in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Drop customerID as it's not needed for analysis
    df = df.drop('customerID', axis=1)
    
    # Convert Churn to binary
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Convert categorical variables to dummy variables
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']
    
    # Create dummy variables for categorical columns
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return X, y

# Model building
def modeling(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE for imbalanced data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Create models dictionary
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        
        # Evaluate model performance
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        report = classification_report(y_test, y_pred)
        
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        elif hasattr(model, 'coef_'):
            feature_importance = pd.Series(model.coef_[0], index=X.columns)
        
        results[name] = {
            'AUC': auc,
            'Report': report,
            'Feature Importance': feature_importance
        }
    
    return results

# Main execution
def main():
    # Load data
    df = load_data()
    
    # Data understanding
    data_understanding(df)
    
    # Exploratory Data Analysis (EDA)
    exploratory_data_analysis(df)
    
    # Data preprocessing
    X, y = preprocessing(df)
    
    # Model building
    results = modeling(X, y)
    
    # Print results
    for model_name, result in results.items():
        print(f"\n{model_name} Results:")
        print(f"AUC: {result['AUC']:.4f}")
        print("\nFeature Importance:")
        if result['Feature Importance'] is not None:
            print(result['Feature Importance'].sort_values(ascending=False).head(10))
        else:
            print("Feature importance not available for this model.")

if __name__ == "__main__":
    main()
