import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

# Install the shap library
!pip install shap
import shap

# Load the Bank Marketing Dataset
data = pd.read_csv('bank-full.csv', sep=';')  # Ensure correct delimiter is used
print(data.head())

# Handle missing values
data = data.dropna()

# Encode categorical features
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Check if all categorical columns exist in the dataset
missing_columns = [col for col in categorical_columns if col not in data.columns]
if missing_columns:
    print(f"Warning: The following columns are missing from the dataset and will be skipped: {missing_columns}")
    categorical_columns = [col for col in categorical_columns if col in data.columns]

# Perform one-hot encoding only on existing columns
data = pd.get_dummies(data, columns=categorical_columns)

# Visualize target variable distribution
sns.countplot(x='y', data=data)
plt.title('Subscription to Term Deposit')
plt.show()

# Prepare features and target
X = data.drop('y', axis=1)
y = data['y'].map({'yes': 1, 'no': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate models
print("Logistic Regression F1-Score:", f1_score(y_test, lr_pred))
print("Random Forest F1-Score:", f1_score(y_test, rf_pred))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr)
plt.title('ROC Curve for Logistic Regression')
plt.show()

# SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Ensure shap_values is a list of arrays for multi-class classification
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)
else:
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

plt.show()

# Explain 5 predictions
for i in range(5):
    if isinstance(shap_values, list):
        shap.plots.force(explainer.expected_value[1], shap_values[1][i], X_test.iloc[i])
    else:
        shap.plots.force(explainer.expected_value, shap_values[i], X_test.iloc[i])
    plt.show()