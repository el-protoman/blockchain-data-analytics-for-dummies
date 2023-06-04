import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X = 50 * np.random.random((30,1))
y = np.where(X<25,0,1)

model = LogisticRegression(solver='lbfgs', random_state=0).fit(X,y.ravel())
X_new = np.linspace(0, 50, 100)
y_new = model.predict(X_new[:, np.newaxis])

ax = plt.axes()
ax.scatter(X, y)
ax.plot(X_new, y_new)

ax.set_xlabel('Hours per week')
ax.set_ylabel('Audition score')

plt.show()

print('Classes: ', model.classes_)
print('Coefficient of determination:', model.score(X, y))
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('confusion matrix: \n', confusion_matrix(y, model.predict(X)))