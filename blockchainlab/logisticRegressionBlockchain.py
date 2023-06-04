import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dfpreprocess = pd.read_csv('products.csv', usecols=['screenType','resolution'])

screenTypeList = dfpreprocess.screenType.unique().tolist()
fscreen = lambda x : screenTypeList.index(x)
converter = {'screenType': fscreen}

df = pd.read_csv('products.csv', usecols=['screenType','reviewRating'], converters=converter)

y = df.reviewRating
X = df.drop('reviewRating', axis=1)

model = LogisticRegression(solver='lbfgs', random_state=0).fit(X,y.ravel())

X_new = np.linspace(40, 70, 60)
y_new = model.predict(X_new[:, np.newaxis])

ax = plt.axes()
ax.scatter(X, y)
ax.plot(X_new, y_new)

ax.set_xlabel('Screen Type')
ax.set_ylabel('review Rating')

plt.show()

print('Coefficient of determination:', model.score(X, y))
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)