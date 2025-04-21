import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.layers import Input, Dense
from keras.models import Model

# Generating a larger dataset
np.random.seed(0)  # For reproducibility
heights = np.random.normal(170, 10, 100)  # Average height 170 cm, standard deviation 10
weights = np.random.normal(65, 15, 100)  # Average weight 65 kg, standard deviation 15

# Combining heights and weights into one dataset
larger_data = np.column_stack((heights, weights))

# Standardizing the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(larger_data)

# Performing PCA to reduce to one dimension
pca = PCA(n_components=1)
data_reduced = pca.fit_transform(data_standardized)

# Visualizing the original and reduced datasets
plt.figure(figsize=(12, 6))

# Original Data
plt.subplot(1, 2, 1)
plt.scatter(larger_data[:, 0], larger_data[:, 1], alpha=0.7)
plt.title('Original Data')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

# PCA Reduced Data
plt.subplot(1, 2, 2)
plt.scatter(data_reduced.flatten(), np.zeros(data_reduced.shape[0]), alpha=0.7)
plt.title('PCA Reduced Data')
plt.xlabel('PCA Component 1')
plt.ylabel('Zeroed Dimension')

plt.tight_layout()
plt.show()





# Autoencoder settings
input_dim = data_standardized.shape[1]  # Number of features (height and weight)
encoding_dim = 1  # Dimension of encoded representation (reduced from 2 to 1)

# Building the autoencoder model
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)

# Separate encoder model
encoder = Model(input_layer, encoded)

# Separate decoder model
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Fit the autoencoder to the standardized data
autoencoder.fit(data_standardized, data_standardized,
                epochs=50,
                batch_size=256,
                shuffle=True,
                verbose=0)

# Use the encoder to encode the data (i.e., reduce its dimensionality)
data_encoded = encoder.predict(data_standardized)

# Visualizing the original and reduced datasets
plt.figure(figsize=(12, 6))

# Original Data
plt.subplot(1, 2, 1)
plt.scatter(larger_data[:, 0], larger_data[:, 1], alpha=0.7)
plt.title('Original Data')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

# Autoencoder Reduced Data
plt.subplot(1, 2, 2)
plt.scatter(data_encoded.flatten(), np.zeros(data_encoded.shape[0]), alpha=0.7)
plt.title('Autoencoder Reduced Data')
plt.xlabel('Encoded Component 1')
plt.ylabel('Zeroed Dimension')

plt.tight_layout()
plt.show()
