#the following code in python shows how to perform linear regression on a sample dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('transfers.csv',usecols=['cost','price','qty','year','month','day'])
X = df[['cost','price','qty','year','month','day']]
y = df['cost']
lm = LinearRegression()
lm.fit(X,y)
print(lm.intercept_)
print(lm.coef_)
print(lm.score(X,y))
print(lm.predict([[100,200,300,2020,1,1]]))
print(lm.predict([[100,200,300,2020,1,2]]))
print(lm.predict([[100,200,300,2020,1,3]]))
print(lm.predict([[100,200,300,2020,1,4]]))
print(lm.predict([[100,200,300,2020,1,5]]))
print(lm.predict([[100,200,300,2020,1,6]]))
print(lm.predict([[100,200,300,2020,1,7]]))
print(lm.predict([[100,200,300,2020,1,8]]))
print(lm.predict([[100,200,300,2020,1,9]]))
print(lm.predict([[100,200,300,2020,1,10]]))
print(lm.predict([[100,200,300,2020,1,11]]))
print(lm.predict([[100,200,300,2020,1,12]]))

