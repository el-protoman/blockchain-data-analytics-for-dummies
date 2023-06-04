# the following code shows how to perform logistic regression on a sample dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('transfers.csv',usecols=['cost','price','qty','year','month','day'])
X = df[['cost','price','qty','year','month','day']]
y = df['cost']
lm = LogisticRegression()
lm.fit(X,y)
print(lm.intercept_)
print(lm.coef_)
print(lm.score(X,y))
print(lm.predict([[100,200,300,2020,1,1]]))
print(lm.predict([[100,200,300,2020,1,2]]))
print(lm.predict([[100,200,300,2020,1,3]]))
print(lm.predict([[100,200,300,2020,1,4]]))
print(lm.predict([[100,200,300,2020,1,5]]))