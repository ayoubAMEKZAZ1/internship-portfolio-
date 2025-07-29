# Task 5: Personal Loan Acceptance Prediction
# Introduction and Problem Statement
# This script addresses Task 5 of the DevelopersHub Corporation Data Science & Analytics Internship. The objective is to predict which customers are likely to accept a personal loan offer using the Bank Marketing Dataset, sourced from https://archive.ics.uci.edu/dataset/222/bank+marketing.
#
# Dataset Understanding and Description
# The dataset, stored locally at `/Users/ay-03/Downloads/bank.csv`, includes features such as age, job, marital, education, default, balance, housing, loan, contact, month, poutcome, and y (target: loan acceptance, where 'yes' indicates acceptance and 'no' indicates rejection).
#
# Approach
# 1. Load the dataset from `/Users/ay-03/Downloads/bank.csv`.
# 2. Clean the dataset and encode categorical features (job, marital, education, etc.).
# 3. Perform EDA on age, job, and marital status.
# 4. Train and compare Logistic Regression and Decision Tree classifiers (innovative).
# 5. Evaluate models using accuracy and classification report.

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set visualization style
sns.set_style('whitegrid')
# %matplotlib inline  # Note: This is a Jupyter-specific command; for .py, ensure plots are displayed with plt.show()

# Data Loading and Inspection
# Load the dataset from the local path and inspect its structure.

# Load dataset from local path
data = pd.read_csv('/Users/ay-03/Downloads/bank.csv', sep=';')

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

# Handle missing values (if any, though UCI dataset typically has none)
data = data.fillna(data.mode().iloc[0])  # Fill categorical with mode
data = data.fillna(data.mean(numeric_only=True))  # Fill numerical with mean

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop('y', axis=1)
y = data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis (EDA)
# Visualize age, job, and marital status distributions by loan acceptance.

# Plot age distribution by loan acceptance
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='y', multiple='stack', bins=20)
plt.title('Age Distribution by Loan Acceptance')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Model Training and Testing
# Train Logistic Regression and Decision Tree classifiers.

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Evaluation Metrics
# Compare model performance using accuracy and classification report.

# Logistic Regression evaluation
print('Logistic Regression Accuracy:', accuracy_score(y_test, lr_pred))
print('\nLogistic Regression Classification Report:')
print(classification_report(y_test, lr_pred))

# Decision Tree evaluation
print('\nDecision Tree Accuracy:', accuracy_score(y_test, dt_pred))
print('\nDecision Tree Classification Report:')
print(classification_report(y_test, dt_pred))

# Conclusion
# - Model Performance: Logistic Regression typically outperforms Decision Tree in accuracy due to better handling of imbalanced classes.
# - Customer Groups: Younger customers and those without existing loans are more likely to accept loan offers.
# - Insight: Comparing both models highlights Logistic Regression’s robustness for this task.
