# pip install yfinance

import yfinance as yf

# Download historical data for AAPL
data = yf.download('AAPL', start='2013-01-01', end='2023-01-01')

# Extracting only the 'Close' column
close_data = data['Close']

# Save the data to a CSV file
close_data.to_csv('AAPL_10_years_close.csv')
