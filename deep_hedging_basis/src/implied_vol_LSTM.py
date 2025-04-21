import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

###############################################################################
# Sequential LSTM model applied to realised volatility of AAPL stock to		  #
# predict impleid volatility. This implied volatiltiy can be compared to the  #
# market's prediction of implied volatility and traded on. If our model 	  #
# predicts a higher implied vol than the mark to market, we buy out of the	  #
# money calls with the corresponding expiries and strikes. We implement a 	  #
# simple version of this for ATM implied vol.								  #
###############################################################################

# Fetch historical data for AAPL
aapl_data = yf.download('AAPL', start='2015-01-01', end='2021-01-01')
close_prices = aapl_data['Close'].values

# Calculate realized volatility (rolling window)
window_size = 10 
returns = np.log(close_prices[1:] / close_prices[:-1])
volatility = pd.Series(returns).rolling(window=window_size).std() * np.sqrt(252)
volatility = volatility.dropna()

# Prepare data for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
volatility = scaler.fit_transform(np.array(volatility).reshape(-1,1))

# Split data into train and test sets
time_step = 3
train_size = int(len(volatility) * 0.75)  # >0.6 yields accurate results
test_size = len(volatility) - train_size
train_data, test_data = volatility[0:train_size,:], volatility[train_size:len(volatility),:1]
print(test_data)

# Reshape into X=t,t+1,t+2,...,t+time_step and Y=t+time_step+1
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))  # 50 neurons in this layer, returns full sequence to next layer
model.add(LSTM(50, return_sequences=False))  # last layer
model.add(Dense(25)) 
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Predicting and inverse transforming the results
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Shift train predictions for plotting
train_predict_plot = np.empty_like(volatility)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict)+time_step, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(volatility)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(time_step*2)+1:len(volatility)-1, :] = test_predict
print(test_predict)

# Plot baseline and predictions
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(volatility), label='Realized Volatility')
plt.plot(train_predict_plot, label='Train Prediction')
plt.plot(test_predict_plot, label='Test Prediction')
plt.title('AAPL Volatility Prediction')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# Save to CSV
predicted_vol_test = test_predict.reshape(-1)
realized_vol_test = y_test.reshape(-1)
data_to_save = {
    "Realized_Volatility": realized_vol_test, 
    "Predicted_Volatility": predicted_vol_test
}
df_to_save = pd.DataFrame(data_to_save)
df_to_save.to_csv('volatility_predictions.csv', index=False)