# pip install scikit-learn

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Gathering data #
# Load data using Pandas
data = pd.read_csv('src/AAPL_10_years_close.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize the Close prices
scaler = MinMaxScaler(feature_range=(0, 1))
data['Normalized_Close'] = scaler.fit_transform(data['Close'].values.reshape(-1,1))




# Feature engineering # 
# Simple Moving Average - 5 days window
data['SMA_5'] = data['Close'].rolling(window=5).mean()

# Exponential Moving Average - 5 days window
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()

# RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Moving Average Convergence Divergence (MACD)
EMA_12 = data['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = EMA_12 - EMA_26
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# On-Balance Volume (OBV)
OBV = [0]
for i in range(1, len(data.Close)):
    if data.Close[i] > data.Close[i-1]:
        OBV.append(OBV[-1] + data.Volume[i])
    elif data.Close[i] < data.Close[i-1]:
        OBV.append(OBV[-1] - data.Volume[i])
    else:
        OBV.append(OBV[-1])
data['OBV'] = OBV
