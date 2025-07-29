# Task 5: Personal Loan Acceptance Prediction

## Objective
Predict which customers are likely to accept a personal loan offer using the Bank Marketing Dataset, sourced from https://archive.ics.uci.edu/dataset/222/bank+marketing, as part of the DevelopersHub Corporation Data Science & Analytics Internship.

## Approach
1. Loaded the dataset from `/Users/ay-03/Downloads/bank.csv`.
2. Handled missing values (if any) and encoded categorical features (job, marital, education, etc.).
3. Performed EDA on age, job, and marital status.
4. Trained and compared Logistic Regression and Decision Tree classifiers (innovative).
5. Evaluated models using accuracy and classification report.

## Results
- **Model Performance**: Logistic Regression outperforms Decision Tree in accuracy due to better handling of imbalanced classes.
- **Customer Groups**: Younger customers and those without existing loans are more likely to accept loan offers.
- **Insight**: Logistic Regression is more robust for this task.

