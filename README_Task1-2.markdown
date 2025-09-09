# Task 1: Term Deposit Subscription Prediction (Bank Marketing)

## Task Objective

Predict whether a bank customer will subscribe to a term deposit as a result of a direct marketing campaign based on phone calls from a Portuguese banking institution.

## How We Tackled the Task

We began by loading the Bank Marketing Dataset (bank.csv with 10% of examples and 17 inputs) from the UCI Machine Learning Repository. After an initial exploration, we cleaned the data and encoded categorical features like job, marital status, and education using one-hot encoding to prepare it for modeling. We then trained two classification models—Logistic Regression and Random Forest—carefully tuning them to fit the data. To assess their performance, we used a Confusion Matrix, F1-Score, and ROC Curve. Finally, we applied SHAP to dive into at least five model predictions, uncovering the factors driving the outcomes.

## Dataset Retrieval

The dataset was retrieved from the following source:

- Bank Marketing Dataset
- https://archive.ics.uci.edu/dataset/222/bank+marketing

## Results and Findings

Random Forest edged out Logistic Regression in F1-Score, suggesting it better captures the complexity of customer behavior. Key influencers like 'duration' (length of the last call) and 'poutcome_success' (success of the previous campaign) stood out in our SHAP analysis, indicating their critical role in predicting subscription likelihood.