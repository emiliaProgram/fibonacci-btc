import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define function to calculate Fibonacci retracement levels
def fibonacci_retracement(high, low):
    levels = {}
    diff = high - low
    levels[0] = high
    levels[0.236] = high - (diff * 0.236)
    levels[0.382] = high - (diff * 0.382)
    levels[0.5] = high - (diff * 0.5)
    levels[0.618] = high - (diff * 0.618)
    levels[0.786] = high - (diff * 0.786)
    levels[1] = low
    return levels

# Define function to create dataset for training RNN
def create_dataset(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Set parameters for RNN model
lookback = 30
n_features = 1
epochs = 50
batch_size = 32

# Download historical data for BTC-AUD from Yahoo Finance API
symbol = "BTC-AUD"
data = yf.download(symbol, start="2015-01-01", end="2023-04-16")

# Calculate Fibonacci retracement levels and plot chart
levels = fibonacci_retracement(data["High"].max(), data["Low"].min())
plt.plot(data["Close"], label="Price")
for key, value in levels.items():
    plt.axhline(value, linestyle="--", color="gray", label=f"Fib {key}")
plt.legend()
plt.title(f"{symbol} - Fibonacci Retracement Levels")
plt.show()