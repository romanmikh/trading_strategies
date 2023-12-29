# pip install scikit-learn

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
print(plt.style.available)


# Gathering data #
# Load data using Pandas
data = pd.read_csv('src/AAPL_10y.csv')

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

# Relative Strength Index (momentum)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss  # relative strength 
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

for i in range(len(data.columns)):
    print(data.columns[i])
print(data)



# Data Visualisation

data_subset = data.tail(300).copy()

# Plot settings
plt.style.use('dark_background')
plt.rc('grid', color='w', linestyle='-', linewidth=1)
plt.rc('axes', facecolor='#EAEAF2', edgecolor='none',
       axisbelow=True, grid=True)
plt.rc('patch', edgecolor='#EAEAF2')
plt.rc('lines', color='C0')
plt.figure(figsize=(20, 15))

# SMA Plot
plt.subplot(5, 1, 1)
plt.plot(data_subset['Close'], label='Close Price', color='blue')
plt.plot(data_subset['SMA_5'], label='5-Day SMA', color='orange')
plt.title('Simple Moving Average (SMA)')
plt.legend()

# EMA Plot
plt.subplot(5, 1, 2)
plt.plot(data_subset['Close'], label='Close Price', color='blue')
plt.plot(data_subset['EMA_5'], label='5-Day EMA', color='green')
plt.title('Exponential Moving Average (EMA)')
plt.legend()

# RSI Plot
plt.subplot(5, 1, 3)
plt.plot(data_subset['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.axhline(30, linestyle='--', alpha=0.5, color='green')
plt.title('Relative Strength Index (RSI)')
plt.legend()

# MACD Plot
plt.subplot(5, 1, 4)
plt.plot(data_subset['MACD'], label='MACD', color='red')
plt.plot(data_subset['Signal_Line'], label='Signal Line', color='black')
plt.title('Moving Average Convergence Divergence (MACD)')
plt.legend()

# OBV Plot
plt.subplot(5, 1, 5)
plt.plot(data_subset['OBV'], label='On-Balance Volume (OBV)', color='brown')
plt.axhline(0, linestyle='--', alpha=0.5, color='red')
plt.title('On-Balance Volume (OBV)')
plt.legend()

# Format Plots
plt.tight_layout()




######## Full Dataset ########


# Plot settings
plt.style.use('dark_background')
plt.rc('grid', color='w', linestyle='-', linewidth=1)
plt.rc('axes', facecolor='#EAEAF2', edgecolor='none',
       axisbelow=True, grid=True)
plt.rc('patch', edgecolor='#EAEAF2')
plt.rc('lines', color='C0')
plt.figure(figsize=(20, 15))

# SMA Plot
plt.subplot(5, 1, 1)
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA_5'], label='5-Day SMA', color='orange')
plt.title('Simple Moving Average (SMA)')
plt.legend()

# EMA Plot
plt.subplot(5, 1, 2)
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['EMA_5'], label='5-Day EMA', color='green')
plt.title('Exponential Moving Average (EMA)')
plt.legend()

# RSI Plot
plt.subplot(5, 1, 3)
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.axhline(30, linestyle='--', alpha=0.5, color='green')
plt.title('Relative Strength Index (RSI)')
plt.legend()

# MACD Plot
plt.subplot(5, 1, 4)
plt.plot(data['MACD'], label='MACD', color='red')
plt.plot(data['Signal_Line'], label='Signal Line', color='black')
plt.title('Moving Average Convergence Divergence (MACD)')
plt.legend()

# OBV Plot
plt.subplot(5, 1, 5)
plt.plot(data['OBV'], label='On-Balance Volume (OBV)', color='brown')
plt.axhline(0, linestyle='--', alpha=0.5, color='red')
plt.title('On-Balance Volume (OBV)')
plt.legend()

# Show the plots
plt.tight_layout()
# plt.show()
