import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

###############################################################################
# Simple cross-moving average trading strategy comparing our predicted vol's  #
# signal with the realised vol moves, calculating PL based on a buy/sell 	  #
# threshold of 10% in this case. The strategy's returns are only a small 	  #
# up for this timeframe - requires further calibration and reconfiguration 	  #
###############################################################################

# Load the data
df = pd.read_csv('volatility_predictions.csv')
realized_vol_test = df['Realized_Volatility'].values
predicted_vol_test = df['Predicted_Volatility'].values

# Define a threshold to trigger trades
vol_change_threshold = 0.1

# Generate trading signals
signals = ['Buy' if pred > real + vol_change_threshold else 'Sell' if pred < real - vol_change_threshold else 'Hold'
           for pred, real in zip(predicted_vol_test, realized_vol_test)]

# Hypothetical trading
trade_results = []
for i in range(len(signals)):
    if signals[i] == 'Buy':
        if i < len(signals) - 1 and realized_vol_test[i + 1] > realized_vol_test[i]:
            trade_results.append(1)  # Win
        else:
            trade_results.append(-1)  # Loss
    elif signals[i] == 'Sell':
        if i < len(signals) - 1 and realized_vol_test[i + 1] < realized_vol_test[i]:
            trade_results.append(1)  # Win
        else:
            trade_results.append(-1)  # Loss
    else:
        trade_results.append(0)  # No trade

# Calculate performance
total_trades = len([t for t in trade_results if t != 0])
successful_trades = len([t for t in trade_results if t == 1])
unsuccessful_trades = len([t for t in trade_results if t == -1])

# Print out the trading performance
print(f"Total Trades: {total_trades}")
print(f"Successful Trades: {successful_trades}")
print(f"Unsuccessful Trades: {unsuccessful_trades}")
print(f"Success Rate: {successful_trades / total_trades * 100:.2f}%" if total_trades > 0 else "No Trades Executed")

# Plotting the results for visualization
plt.figure(figsize=(12, 6))
plt.plot(realized_vol_test, label='Realized Volatility', color='blue')
plt.plot(predicted_vol_test, label='Predicted Volatility', color='orange')

# Correcting the scatter plot for winning trades
winning_indices = [i for i in range(len(trade_results)) if trade_results[i] == 1]
winning_trades = [realized_vol_test[i] for i in winning_indices]
plt.scatter(winning_indices, winning_trades, marker='^', color='green', label='Winning Trade')

# Correcting the scatter plot for losing trades
losing_indices = [i for i in range(len(trade_results)) if trade_results[i] == -1]
losing_trades = [realized_vol_test[i] for i in losing_indices]
plt.scatter(losing_indices, losing_trades, marker='v', color='red', label='Losing Trade')

plt.title('Implied Volatility and Trading Signals')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()







# Simple Moving Average Crossover Strategy
def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Settings for the trading strategy
window_size = 10  
investment_per_trade = 10  # $10 of vega per trade
vol_change_to_dollar = 100  # 0.1% vol change equals $1

# Calculating the moving average of the predicted volatility
predicted_vol_series = pd.Series(predicted_vol_test)
moving_avg_predicted_vol = moving_average(predicted_vol_series, window_size)

# Initialize variables for the strategy
total_position = 0  # Total vega position
pnl = [0]  # Starting with no profit or loss

# Execute trading strategy
for i in range(window_size, len(predicted_vol_test)):
    if predicted_vol_test[i] > moving_avg_predicted_vol[i] and total_position <= 0:
        # Buy signal
        total_position += investment_per_trade
    elif predicted_vol_test[i] < moving_avg_predicted_vol[i] and total_position >= 0:
        # Sell signal
        total_position -= investment_per_trade
    
    # Calculate P&L based on the actual realized volatility change
    if i < len(predicted_vol_test) - 1:
        vol_change_percent = (realized_vol_test[i + 1] - realized_vol_test[i]) / realized_vol_test[i]
        trade_pnl = vol_change_percent * vol_change_to_dollar * total_position
        pnl.append(pnl[-1] + trade_pnl)

# Plotting the results for visualization
plt.figure(figsize=(12, 6))
plt.plot(realized_vol_test, label='Realized Volatility', color='blue')
plt.plot(predicted_vol_test, label='Predicted Volatility', color='orange')
plt.plot(moving_avg_predicted_vol, label='Moving Average Predicted Volatility', color='green')

# Adding P&L to the plot
plt.plot(pnl, label='Cumulative P&L', color='purple', marker='o', linestyle='--')

plt.title('Volatility Prediction, Trading Strategy, and Cumulative P&L')
plt.xlabel('Time (Days)')
plt.ylabel('Volatility / P&L')
plt.legend()
plt.show()






