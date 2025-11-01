# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 01.11.2025
### Reg No:212223240068

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_csv('train.csv')

# Choose a numeric feature to simulate as a time-series
target_variable = 'ram'

# Create an index to act as a time axis
data['Index'] = pd.RangeIndex(start=1, stop=len(data)+1)
data.set_index('Index', inplace=True)

def arima_model(data, target_variable, order=(5,1,0)):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='green')
    plt.xlabel('Index')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Run ARIMA on one numeric column
arima_model(data, target_variable, order=(5,1,0))
```

### OUTPUT:

<img width="805" height="531" alt="image" src="https://github.com/user-attachments/assets/e3a01d56-61d1-42ee-9d32-3bcdcb5247eb" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
