import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pd.read_csv('transfers.csv',usecols=['cost','price','qty','year','month','day'])
scatter_matrix(df, alpha=0.2, figsize=(6, 6))
plt.show()