# Task 2: Data Cleaning and Handling Missing Values

## Objective
Clean the Loan Default dataset, sourced from https://www.kaggle.com/datasets/nikhil1e9/loan-default, and handle missing values to prepare it for analysis, as part of the DevelopersHub Corporation Data Science & Analytics Internship.

## Approach
1. Loaded the dataset from `/Users/ay-03/Downloads/Loan_Default.csv`.
2. Inspected dataset structure (shape, columns, data types, missing values).
3. Handled missing values using:
   - KNN imputation for numerical features (Income, Age, Experience, etc.) (innovative).
   - Mode imputation for categorical features (Married/Single, House_Ownership, etc.).
4. Visualized income distributions before and after imputation to confirm data integrity.
5. Evaluated data quality and summarized findings.

## Results
- **Data Quality**: Successfully handled all missing values in numerical and categorical features.
- **Data Integrity**: KNN imputation preserved the income distribution, as shown in histograms.
- **Insights**: The cleaned dataset is ready for downstream tasks like loan default prediction.

