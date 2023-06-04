# https://bscscan.com/address/0xfbffeccb8ec2d959b15699ef80d035c5f5a7bf50
# contract address:
# 0xfbffeccb8ec2d959b15699ef80d035c5f5a7bf50

# Call the BSC API for transaction data
# https://api.bscscan.com/api?module=account&action=txlist&address=0xfbffeccb8ec2d959b15699ef80d035c5f5a7bf50&startblock=0&endblock=99999999&sort=asc&apikey=YourApiKeyToken

import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

API_KEY = '7JW9BE6CZW7PJHI5D45M2Y7BV2V63TTKU6'
CONTRACT_ADDRESS = '0xfbffeccb8ec2d959b15699ef80d035c5f5a7bf50'

def get_transactions(contract_address, api_key):
    url = f"https://api.bscscan.com/api?module=account&action=tokentx&contractaddress={contract_address}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == '1':
        # if the response is successful, convert the result into a DataFrame
        df = pd.DataFrame(data['result'])
        return df
    else:
        # if the response is not successful, print the error message and return None
        print(f"Error fetching data: {data['message']}")
        return None

#df = get_transactions(CONTRACT_ADDRESS, API_KEY)
# print(df.head(10))
# print(df.dtypes)

file_path = 'export-SGO.csv'
df = pd.read_csv(file_path)
print(df.head(10))
print(df['DateTime'].head(10))
print(df.dtypes)

# Convert the 'DateTime' column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%m/%d/%Y %H:%M")

# Set the 'DateTime' column as the index
df.set_index('DateTime', inplace=True)

# Check for missing values
print(df.isnull().sum())

# Fill missing values (if any)
# df.fillna(method='ffill', inplace=True)  # forward fill
# df.fillna(method='bfill', inplace=True)  # backward fill
# df.interpolate(method='linear', inplace=True)  # linear interpolation

# Resample the data to daily frequency (if needed)
# df = df.resample('D').mean()

# Print the first 10 rows of the DataFrame
print(df.head(10))

# EDA
# Plot the time series
df['TxnFee(BNB)'].plot()
plt.title('TxnFee over time')
plt.show()

# # Replace 'Error(0)' with NaN
# df['Historical $Price/BNB'] = df['Historical $Price/BNB'].replace('Error(0)', np.nan)

# # Convert the column to float
# df['Historical $Price/BNB'] = df['Historical $Price/BNB'].astype(float)

# # Fill missing values with the median
# df['Historical $Price/BNB'] = df['Historical $Price/BNB'].fillna(0, inplace=True)
# df = df.fillna(value={'Historical $Price/BNB': 0})

# Convert the column to integer
df['Historical $Price/BNB'] = df['Historical $Price/BNB'].astype(int)


df['Historical $Price/BNB'].plot()
plt.title('Historical $Price/BNB over time')
plt.show()

# Count the occurrences of each unique value in the 'Method' column
method_counts = df['Method'].value_counts()

# Create a bar plot
method_counts.plot(kind='bar')

# Set the title and labels
plt.title('Counts of Different Methods')
plt.ylabel('Count')
plt.xlabel('Method')

# Show the plot
plt.show()

df['Price_Fee_Interaction'] = df['Historical $Price/BNB'] * df['TxnFee(BNB)']
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Price_Fee_Interaction'], label='Price_Fee_Interaction')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Price_Fee_Interaction Over Time')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.hist(df['Price_Fee_Interaction'], bins=50, alpha=0.5, label='Price_Fee_Interaction')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Price_Fee_Interaction')
plt.legend()
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Specify the parameters for the ARIMA model
p = 1
d = 1
q = 3

# Create the ARIMA model
model = ARIMA(df['Historical $Price/BNB'], order=(p,d,q))

# Fit the model
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=10)  # forecast the next 10 steps
