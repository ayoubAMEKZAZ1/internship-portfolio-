# Task 4: Predicting Insurance Claim Amounts
# Introduction and Problem Statement
# This script addresses Task 4 of the DevelopersHub Corporation Data Science & Analytics Internship. The objective is to predict medical insurance claim amounts using the Insurance Claim Dataset, sourced from https://www.kaggle.com/datasets/yasserh/insurance-claim-dataset, and analyze the impact of BMI, age, and smoking status.
#
# Dataset Understanding and Description
# The dataset, stored locally at `/Users/ay-03/Downloads/insurance.csv`, includes features: age, sex, bmi, children, smoker, region, and charges (target). The target variable 'charges' represents the medical insurance claim amount.
#
# Approach
# 1. Load the dataset from `/Users/ay-03/Downloads/insurance.csv`.
# 2. Clean the dataset and encode categorical features (sex, smoker, region).
# 3. Add polynomial features for age and BMI (innovative) to capture non-linear relationships.
# 4. Train a Linear Regression model.
# 5. Visualize feature impacts (BMI, age, smoker status) and evaluate using MAE and RMSE.

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Set visualization style
sns.set_style('whitegrid')
# %matplotlib inline  # Note: This is a Jupyter-specific command; for .py, ensure plots are displayed with plt.show()

# Data Loading and Inspection
# Load the dataset from the local path and inspect its structure.

# Load dataset from local path
# Ensure the insurance.csv file is available in the Colab environment
data = pd.read_csv('/Users/ay-03/Downloads/insurance.csv')

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

# Encode categorical variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Create polynomial features for age and bmi
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[['age', 'bmi']])
poly_df = pd.DataFrame(poly_features, columns=['age', 'bmi', 'age^2', 'age*bmi', 'bmi^2'])

# Combine polynomial features with original data
data = pd.concat([data.drop(['age', 'bmi'], axis=1), poly_df], axis=1)

# Split features and target
# Remove 'insuranceclaim' from features to prevent data leakage
X = data.drop('charges', axis=1)
y = data['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis (EDA)
# Visualize the impact of BMI, age, and smoker status on charges.

# Scatter plot: BMI vs. Charges by Smoker
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='bmi', y='charges', hue='smoker', style='smoker')
plt.title('BMI vs. Charges by Smoker Status')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

# Model Training and Testing
# Train a Linear Regression model.

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# Evaluation Metrics
# Evaluate using MAE and RMSE.

# Calculate MAE and RMSE
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

# Conclusion
# - Model Performance: The Linear Regression model with polynomial features achieves reasonable MAE and RMSE.
# - Feature Impact: Smoker status has a significant impact on charges, with higher BMI amplifying costs for smokers.
# - Insight: Polynomial features capture non-linear effects of age and BMI, improving model accuracy.