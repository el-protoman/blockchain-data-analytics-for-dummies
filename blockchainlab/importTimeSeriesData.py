import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Dataset source: https://archive.ics.uci.edu/ml/datasets

def convert_currency(val):
    new_val = val.replace(',','').replace('$','')
    return float(new_val)

# Import data
#df = pd.read_csv('dow_jones_index.data', usecols=['stock','date','close'], index_col='date', header=0, parse_dates=True)


raw_df = pd.read_csv('dow_jones_index.data', usecols=['stock','date','close'], index_col='date', header=0, parse_dates=True, converters={'close':convert_currency})
df = raw_df[(raw_df['stock'] == 'AXP')]
del df['stock']

print(df.head())
print(df.dtypes)

df_ar = pd.plotting.autocorrelation_plot(df)
diff = df.diff(1)
plt.plot(diff)
diff2 = df.diff(2)
plt.plot(diff2)

diff3 = df.diff(3)
plt.plot(diff3)

df.plot()

rollmean = df.rolling(6).mean()
plt.plot(rollmean, color='red', label='Rolling Mean')

plt.show()

# 1,0,3 ARIMA Model
model = ARIMA(df.close, order=(1,0,3))
model_fit = model.fit()
plt.plot(model_fit.fittedvalues, color='green')
print(model_fit.summary())