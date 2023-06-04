import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

API_KEY = '7JW9BE6CZW7PJHI5D45M2Y7BV2V63TTKU6'
CONTRACT_ADDRESS = '0xfbffeccb8ec2d959b15699ef80d035c5f5a7bf50'
file_path = 'export-SGO.csv'

# Load the data
df = pd.read_csv(file_path)
df = df[['DateTime', 'TxnFee(BNB)', 'Historical $Price/BNB', 'Blockno']]

# Convert the 'DateTime' column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%m/%d/%Y %H:%M")

# Set the 'DateTime' column as the index
df.set_index('DateTime', inplace=True)

# Replace 'Error(0)' with NaN
df['Historical $Price/BNB'] = df['Historical $Price/BNB'].replace('Error(0)', np.nan)

# Convert the column to float
df['Historical $Price/BNB'] = df['Historical $Price/BNB'].astype(float)

# Fill missing values with the median
df['Historical $Price/BNB'].fillna(df['Historical $Price/BNB'].median(), inplace=True)

# Create a new feature
df['Price_Fee_Interaction'] = df['Historical $Price/BNB'] * df['TxnFee(BNB)']

# Resample the data to daily frequency (if needed)
df = df.resample('D').mean()

# Plotting
df['TxnFee(BNB)'].plot()
plt.title('TxnFee(BNB) over Time')
plt.show()

df['Historical $Price/BNB'].plot()
plt.title('Historical $Price/BNB over Time')
plt.show()

df['Price_Fee_Interaction'].plot()
plt.title('Price_Fee_Interaction over Time')
plt.show()

# ARIMA model
model = ARIMA(df['Historical $Price/BNB'].astype(float), order=(1,1,1))  # Changed the order to (1,1,1) to make the model stationary.
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=10)

# Print predictions
print(predictions)
