# Task 4: Loan Default Risk with Business Cost Optimization

## Task Objective
Predict the likelihood of a loan default and optimize the decision threshold based on cost-benefit analysis using the Home Credit Default Risk Dataset.

## How We Tackled the Task
We began with the [Home Credit Default Risk Dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_test.csv) and cleaned it by addressing missing values. After preprocessing, we trained Logistic Regression and CatBoost models to predict defaults. We assigned business cost values—100 for false positives and 1000 for false negatives—and experimented with different thresholds to minimize total cost. This process helped us fine-tune the models for real-world application.

## Dataset Retrieval
The dataset was retrieved from the following source:  
- [Home Credit Default Risk Dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_test.csv)

## Results and Findings
Both models performed similarly, but adjusting the threshold significantly reduced overall cost. The optimal threshold balanced the impact of false positives and negatives, with feature importance analysis revealing key risk factors driving default predictions.