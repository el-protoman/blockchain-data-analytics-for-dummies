import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('products.csv', usecols=['screenSize', 'msrp'])

y = df.msrp
X = df.drop('msrp', axis=1)

model = LinearRegression().fit(X, y)

X_new = np.linspace(40, 70, 60)
y_new = model.predict(X_new[:, np.newaxis])

ax = plt.axes()
ax.scatter(X, y)
ax.plot(X_new, y_new)

ax.set_xlabel('Screen Size')
ax.set_ylabel('MSRP')

plt.show()
print('Coefficient of determination', model.score(X, y))
print('Coefficients', model.coef_)
print('Coefficientss', model.intercept_)