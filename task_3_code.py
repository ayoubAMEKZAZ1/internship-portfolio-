import pandas as pd

# Load the Household Power Consumption Dataset
data = pd.read_csv('household_power_consumption.csv', sep=';', parse_dates={'datetime': ['Date', 'Time']}, na_values=['?'])
print(data.head())

# Handle missing values and resample
data = data.dropna()
data = data.resample('D', on='datetime').mean()

# Ensure no NaN, infinity, or excessively large values in the target column
data = data[~data['Global_active_power'].isnull()]
data = data[~data['Global_active_power'].isin([float('inf'), float('-inf')])]
data = data[data['Global_active_power'] < 1e6]  # Arbitrary large value threshold

# Engineer time-based features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek

# Install missing libraries
!pip install statsmodels
!pip install prophet
!pip install xgboost

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prepare data for Prophet
prophet_data = data.reset_index().rename(columns={'datetime': 'ds', 'Global_active_power': 'y'})
model_prophet = Prophet()
model_prophet.fit(prophet_data)

# ARIMA and XGBoost models (simplified example)
model_arima = ARIMA(data['Global_active_power'], order=(5,1,0)).fit()
model_xgb = XGBRegressor().fit(data[['hour', 'day_of_week']], data['Global_active_power'])

# Forecast
forecast_prophet = model_prophet.predict(prophet_data.tail(30))
forecast_arima = model_arima.forecast(steps=30)
forecast_xgb = model_xgb.predict(data[['hour', 'day_of_week']].tail(30))

# Evaluate
print("Prophet MAE:", mean_absolute_error(data['Global_active_power'].tail(30), forecast_prophet['yhat']))
print("ARIMA MAE:", mean_absolute_error(data['Global_active_power'].tail(30), forecast_arima))
print("XGBoost MAE:", mean_absolute_error(data['Global_active_power'].tail(30), forecast_xgb))

# Visualizations
import matplotlib.pyplot as plt
plt.plot(data.index[-60:], data['Global_active_power'].tail(60), label='Actual')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Prophet Forecast')
plt.title('Actual vs Forecasted Energy Usage')
plt.legend()
plt.show()