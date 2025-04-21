import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV

# Step 1: Data Collection
# Fetch historical stock data using yfinance
ticker_symbol = 'AAPL'  # Example: Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2021-01-01')

# Step 2: Data Preprocessing
df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Vanilla Autoencoder Model Creation

# Model parameters
input_dim = scaled_data.shape[1]  # number of features
encoding_dim = 3  # dimension of encoded data

# Encoder
input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

# Autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Display the autoencoder structure
autoencoder.summary()

# Step 4: Model Training
# For demonstration, using a small number of epochs
autoencoder.fit(scaled_data, scaled_data, epochs=10, batch_size=32, shuffle=True)





# Step 1: Visualize the Original and Reconstructed Data
reconstructed_data = autoencoder.predict(scaled_data)

# Plotting the original and reconstructed data for comparison
plt.figure(figsize=(15, 6))
plt.plot(scaled_data[:, 3], label='Original Close Prices')  # Assuming 3rd index is 'Close'
plt.plot(reconstructed_data[:, 3], label='Reconstructed Close Prices')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.show()

# Step 2: Feature Extraction
encoded_features = autoencoder.predict(scaled_data)

# Step 3: Visualize the Encoded Features
# This is a simple plot, more sophisticated methods might be required for deeper analysis
plt.figure(figsize=(15, 6))
plt.plot(encoded_features)
plt.title('Encoded Features Over Time')
plt.show()

# Step 4: Formulate a Basic Trading Strategy
# Example: A very simple strategy based on the moving average of encoded features
moving_average = pd.Series(encoded_features[:, 0]).rolling(window=5).mean()  # Simple moving average

# Generating buy/sell signals
buy_signals = moving_average > encoded_features[:, 0]
sell_signals = moving_average < encoded_features[:, 0]

# Step 5: Visualization
plt.figure(figsize=(15, 6))
plt.plot(df['Close'], label='Close Prices') # Plotting the actual close prices
plt.scatter(df.index[buy_signals], df['Close'][buy_signals], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(df.index[sell_signals], df['Close'][sell_signals], marker='v', color='r', label='Sell Signal', alpha=1)
plt.title('Trading Signals on Close Prices')
plt.legend()
plt.show()





# Hyperparaneter optimisation
def create_autoencoder(encoding_dim=3, activation='relu', learning_rate=0.001):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder_layer = Dense(encoding_dim, activation=activation)(input_layer)

    # Decoder
    decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

    # Autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    return autoencoder

# Define the grid search parameters
param_grid = {
    'encoding_dim': [2, 3, 5],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Create a KerasRegressor wrapper for our model
model = KerasRegressor(build_fn=create_autoencoder, epochs=10, batch_size=32, verbose=0)

# Create GridSearchCV and fit the data
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(scaled_data, scaled_data)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
results = grid_result.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print("%f with: %r" % (mean_score, params))
