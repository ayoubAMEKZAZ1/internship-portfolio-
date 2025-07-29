# Task 3: Customer Churn Prediction
# Introduction and Problem Statement
# This script addresses Task 3 of the DevelopersHub Corporation Data Science & Analytics Internship. The objective is to predict customers likely to leave a bank using the Bank Customer Attrition Insights dataset, sourced from https://www.kaggle.com/datasets/marusagar/bank-customer-attrition-insights, and analyze feature importance.
#
# Dataset Understanding and Description
# The dataset, stored locally at `/Users/ay-03/Downloads/BankChurn.csv`, includes features such as CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, and Attrition (target, indicating churn). The target variable 'Attrition' indicates whether a customer has left the bank (1) or stayed (0).
#
# Approach
# 1. Load the dataset from `/Users/ay-03/Downloads/BankChurn.csv`.
# 2. Clean the dataset and encode categorical features (Geography, Gender).
# 3. Train a Random Forest classifier.
# 4. Use SHAP values (innovative) for feature importance analysis.
# 5. Evaluate model performance with accuracy, confusion matrix, and classification report.

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Install SHAP module
import sys
!{sys.executable} -m pip install shap
import shap

# Set visualization style
sns.set_style('whitegrid')

# Data Loading and Inspection
# Load the dataset from the local path and inspect its structure.

# Load dataset from local path
data = pd.read_csv('/Users/ay-03/Downloads/BankChurn.csv')

# Display shape, columns, and first few rows
print('Dataset Shape:', data.shape)
print('\nColumn Names:', data.columns.tolist())
print('\nFirst 5 Rows:')
print(data.head())

# Check missing values
print('\nMissing Values:')
print(data.isnull().sum())

# Check data types
print('\nData Types:')
print(data.dtypes)

# Data Cleaning and Preparation
# Handle missing values (if any) and encode categorical features.

# Handle missing values
data = data.fillna(data.mode().iloc[0])  # Fill categorical with mode
data = data.fillna(data.mean(numeric_only=True))  # Fill numerical with mean

# Drop irrelevant columns
data = data.drop(['CustomerId', 'Surname'], axis=1, errors='ignore')

# Fixing the target column name issue
# Check if 'Exited' column exists and rename it to 'Attrition'
if 'Exited' in data.columns:
    data.rename(columns={'Exited': 'Attrition'}, inplace=True)
else:
    raise KeyError("The target column 'Exited' is not found in the dataset.")

# Encode categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Card Type'] = le.fit_transform(data['Card Type'])  # Encode 'Card Type' to handle 'DIAMOND' issue

# One-hot encode Geography
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# Split features and target (Attrition as target)
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis (EDA)
# Visualize age distribution by churn status to understand patterns.

# Plot age distribution by churn
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Attrition', multiple='stack', bins=20)
plt.title('Age Distribution by Churn Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Model Training and Testing
# Train a Random Forest classifier.

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluation Metrics
# Evaluate model performance using accuracy, confusion matrix, and classification report.

# Calculate accuracy
print('Accuracy:', accuracy_score(y_test, y_pred))

# Confusion matrix
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Feature Importance Analysis
# Use SHAP values for interpretable feature importance.

# Initialize SHAP explainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values.values, X_test, plot_type='bar')
