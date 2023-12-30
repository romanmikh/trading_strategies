# Update NVIDIA graphics drivers https://www.nvidia.com/download/index.aspx
# Install CUDA ToolKit https://developer.nvidia.com/cuda-downloads
# Install CUDA NN Compiler https://developer.nvidia.com/rdp/cudnn-download
# pip install tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load data using Pandas
data = pd.read_csv('src/AAPL_10y.csv')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Define how many days of data the model should consider & split data
sequence_length = 60
train_size = int(len(scaled_data) * 0.8)  # Example: 80% of data for training

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - sequence_length:]  # Overlap last part of training data

# Create Training and Test Generators
train_generator = TimeseriesGenerator(train_data, train_data, length=sequence_length, batch_size=1)
test_generator = TimeseriesGenerator(test_data, test_data, length=sequence_length, batch_size=1)

# LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_generator, epochs=25, batch_size=32)

# Evaluate the model
evaluation_metrics = model.evaluate(test_generator)
print(f"Test Loss: {evaluation_metrics}")

# Generate predictions for the test set
predictions = model.predict(test_generator)

# Inverse transform the predictions
predicted_prices = scaler.inverse_transform(predictions)

# Actual test data for comparison
actual_test_data = scaler.inverse_transform(test_data[sequence_length:])

# Plotting
plt.plot(actual_test_data, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
