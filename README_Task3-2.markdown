# Task 3: Energy Consumption Time Series Forecasting

## Task Objective
Forecast short-term household energy usage using historical time-based patterns from the Household Power Consumption Dataset.

## How We Tackled the Task
We kicked off by loading the Household Power Consumption Dataset (sourced from UCI Machine Learning Repository) and parsed it into daily averages after resampling. We enriched the data with time-based features like hour of day and weekday/weekend indicators. Next, we compared the performance of ARIMA, Prophet, and XGBoost models, evaluating them with Mean Absolute Error (MAE). To bring the results to life, we plotted actual versus forecasted energy usage, making the trends easy to spot.

## Dataset Retrieval
The dataset was retrieved from the following source:  
- [Household Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

## Results and Findings
The Prophet model emerged as the top performer with the lowest MAE, showing its strength in capturing time-based patterns. Adding features like hour and day of week boosted all models’ accuracy. Our plots revealed a tight match between actual and forecasted values, with Prophet leading the way.