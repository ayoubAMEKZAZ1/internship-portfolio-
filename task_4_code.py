import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the Home Credit Default Risk Dataset
data1 = pd.read_csv('application_train.csv')
data2 = pd.read_csv('application_test.csv')
combined_data = pd.concat([data1, data2], ignore_index=True)

# Handle missing values
# Separate numeric and non-numeric columns
numeric_cols = combined_data.select_dtypes(include=['number']).columns
non_numeric_cols = combined_data.select_dtypes(exclude=['number']).columns

# Impute numeric columns with mean
imputer = SimpleImputer(strategy='mean')
combined_data[numeric_cols] = imputer.fit_transform(combined_data[numeric_cols])

# Impute non-numeric columns with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
combined_data[non_numeric_cols] = imputer.fit_transform(combined_data[non_numeric_cols])

# Convert categorical columns to numeric using one-hot encoding
combined_data = pd.get_dummies(combined_data, drop_first=True)

# Ensure the target column is binary (0 or 1)
combined_data['TARGET'] = combined_data['TARGET'].astype(int)

# Prepare features and target
X = combined_data.drop('TARGET', axis=1)
y = combined_data['TARGET']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict_proba(X_test)[:, 1]

cb_model = CatBoostClassifier(verbose=0)
cb_model.fit(X_train, y_train)
cb_pred = cb_model.predict_proba(X_test)[:, 1]

# Define business cost values
false_positive_cost = 100
false_negative_cost = 1000

def calculate_cost(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1] * false_positive_cost
    fn = cm[1, 0] * false_negative_cost
    return fp + fn

# Adjust the model threshold to minimize total business cost
best_threshold = min(range(1, 100), key=lambda x: calculate_cost(y_test, lr_pred, x/100))
print("Optimal Threshold:", best_threshold/100)

# Visualizations
plt.hist(lr_pred, bins=50)
plt.title('Predicted Probabilities Distribution')
plt.show()