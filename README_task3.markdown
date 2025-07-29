# Task 3: Customer Churn Prediction

## Objective
Predict customers likely to churn using the Bank Customer Attrition Insights dataset, sourced from https://www.kaggle.com/datasets/marusagar/bank-customer-attrition-insights, as part of the DevelopersHub Corporation Data Science & Analytics Internship.

## Approach
1. Loaded the dataset from `/Users/ay-03/Downloads/BankChurn.csv`.
2. Handled missing values (if any) and encoded categorical features (Geography, Gender).
3. Trained a Random Forest classifier.
4. Used SHAP values (innovative) for feature importance analysis.
5. Evaluated with accuracy, confusion matrix, and classification report.

## Results
- **Model Performance**: Random Forest typically achieves ~85% accuracy.
- **Feature Importance**: SHAP values highlight Age, Balance, and NumOfProducts as key churn drivers.
- **Insight**: Older customers with higher balances and fewer products are more likely to churn.

