import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data
x = 50 * np.random.random((30,1))
# y = 0.5 * x + 1.0 + np.random.normal(size=x.shape)
y = 2 * x + 10.0 * np.random.normal(size=x.shape)

model = LinearRegression().fit(x,y)

x_new = np.linspace(0, 50, 100)
y_new = model.predict(x_new[:, np.newaxis])

ax = plt.axes()
ax.scatter(x, y)
ax.plot(x_new, y_new)

ax.set_xlabel('Hours per week')
ax.set_ylabel('Audition score')

plt.show()