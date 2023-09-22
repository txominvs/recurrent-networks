import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
timesteps = 100
x = np.linspace(0, 10, timesteps)
y = np.sin(x) + 0.1 * np.random.randn(timesteps)

# Split the data into training and testing sets
train_ratio = 0.8
train_size = int(train_ratio * timesteps)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Create sequences for the LSTM
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i : i + sequence_length]
        target = data[i + sequence_length]
        sequences.append((seq, target))
    return np.array(sequences)

sequence_length = 10
train_sequences = create_sequences(y_train, sequence_length)
test_sequences = create_sequences(y_test, sequence_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Reshape the data for training
x_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]
x_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1]

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))

# Make predictions
predicted_values = model.predict(x_test)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(train_size + sequence_length, timesteps), y_test, marker='o', label='True')
plt.plot(np.arange(train_size + sequence_length, timesteps), predicted_values, marker='x', label='Predicted')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('LSTM Time Series Prediction')
plt.show()