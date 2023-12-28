# pip install scikit-learn

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data using Pandas
data = pd.read_csv('AAPL_10_years_close.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize the Close prices
scaler = MinMaxScaler(feature_range=(0, 1))
data['Normalized_Close'] = scaler.fit_transform(data['Close'].values.reshape(-1,1))

print(data)