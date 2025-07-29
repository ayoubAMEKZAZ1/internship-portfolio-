# Task 4: Predicting Insurance Claim Amounts

## Objective
Predict medical insurance claim amounts using the Insurance Claim Dataset, sourced from https://www.kaggle.com/datasets/yasserh/insurance-claim-dataset, as part of the DevelopersHub Corporation Data Science & Analytics Internship.

## Approach
1. Loaded the dataset from `/Users/ay-03/Downloads/insurance.csv`.
2. Handled missing values (if any) and encoded categorical features (sex, smoker, region).
3. Added polynomial features for age and BMI (innovative) to capture non-linear relationships.
4. Trained a Linear Regression model.
5. Visualized impacts of BMI, age, and smoker status; evaluated with MAE and RMSE.

## Results
- **Model Performance**: Linear Regression with polynomial features achieves reasonable MAE and RMSE.
- **Feature Impact**: Smoker status and higher BMI significantly increase charges.
- **Insight**: Polynomial features improve model accuracy by capturing non-linear effects.

