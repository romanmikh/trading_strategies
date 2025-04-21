# pip install yfinance

import yfinance as yf

# Download historical data for AAPL
data = yf.download('AAPL', start='2013-01-01', end='2023-01-01')

# Save the data to a CSV file
data.to_csv('src/AAPL_10y.csv')