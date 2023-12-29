# Update NVIDIA graphics drivers https://www.nvidia.com/download/index.aspx
# Install CUDA ToolKit https://developer.nvidia.com/cuda-downloads
# pip install tensorflow

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load data using Pandas
data = pd.read_csv('src/AAPL_10y.csv')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Define how many days of data the model should consider
sequence_length = 60

# Generate time series sequences
generator = TimeseriesGenerator(scaled_data, scaled_data, length=sequence_length, batch_size=1)
