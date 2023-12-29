<p align="center"><i>This city was once a happy, peaceful place...</i></p>
<p align="center"><i>Until one day, a powerful secret criminal organization took over.</i></p>
<p align="center"><i>This vicious syndicate soon had control of the government</i></p>
<p align="center"><i>and even the police force. The city has become a center of</i></p>
<p align="center"><i>violence and crime where no one is safe.</i></p>
<p align="center"><i>Amid this turmoil, a group of determined young traders</i></p>
<p align="center"><i>has sworn to clean up the city. Among them are</i></p>
<p align="center"><i><b>Samius Taylorium, Professor X Meyer</b> and <b>Romano Mikhalabongo</b>.</i></p>
<p align="center"><i>They are willing to risk anything... Even their lives... On the...</i></p>

<p align="center"><i><b>Streets of Leverage</b></i></p>



<br><br><br><br>



<b>Plan of action</b>

1. Team meeting

2. Gather Data
Sources: Yahoo Finance, Alpha Vantage, or Quandl to get historical stock price data.
Data Range: 10y+

3. Preprocess the Data
Cleaning: Handle missing values, outliers, and anomalies.<br>
Normalization: Stock price data should be normalized for better performance of neural networks.<br>
Feature Engineering to capture trends:<br>
    - Simple Moving Averages<br>
    - Exponential Moving Averages<br>
    - Relative Strength Index (https://en.wikipedia.org/wiki/Relative_strength_index)<br>
    - Moving Average Convergence DIvergence (https://en.wikipedia.org/wiki/MACD) <br>
    - On-Balance Volume (https://en.wikipedia.org/wiki/On-balance_volume) <br>

4. Choose an AI Model
Time Series Models: LSTM (Long Short Term Memory) networks are popular for time series prediction. Other options include ARIMA, GARCH, or more complex neural networks like Transformers.<br>
Machine Learning Models: Regression models, Random Forest, Gradient Boosting, etc.

5. Train the Model
Split the Data: Divide your data into training, validation, and test sets.<br>
Training: Use the training set to train your model.<br>
Validation: Use the validation set to fine-tune hyperparameters and avoid overfitting.<br>

6. Evaluate the Model
Metrics: Use metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), or others suitable for regression problems.<br>
Backtesting: Simulate how the model would perform on historical data.

7. Implementation
Real-Time Data: For real-world application, you need to integrate real-time data feeds.<br>
Automation: Automate the process of data fetching, prediction, and trading.

Libraries: Pandas for data manipulation, NumPy for numerical operations, Matplotlib/Seaborn for visualization, scikit-learn for basic ML models, TensorFlow/Keras or PyTorch for neural networks.



<br><br><br><br>



<b>Notes:</b> <br><br>
RSI:
> Ratio of up momentum VS down momentum<br>
> \>70 --> overbought + will fall<br>
> <30 --> oversold + will rise<br>


MACD:
> MACD = 0 --> long term averages converge<br>
> 1) MACD = 0 upwards & above signal --> bullish, short term momentum rising faster than long term<br>
> 2) MACD = 0 upwards & below signal --> weak bullish, convergence of momenta<br>
> 3) MACD = 0 downwards & above signal --> weak bearish, convergence of momenta<br>
> 4) MACD = 0 downwards & below signal --> bearish, short term momentum falling faster than long term<br>


OBV:
> Assumes volume preceeds price movement<br>
> price up --> cumulatively add volume<br>
> price down --> cumulatively subtract volume<br>
> 1) price rising & OBV *gradient* decreasing --> bearish reversal likely<br>
> 2) price falling & OBV *gradient* increasing --> bullish reversal likely<br>

