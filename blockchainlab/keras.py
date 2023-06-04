from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
#load your data
mydata = loadtxt('export-SGO.csv', delimiter=',')
# partition data into input (X) and output (y)
X = mydata[:,0:8]
y = mydata[:,8]
# define the keras model
nn = Sequential()
nn.add(Dense(12, input_dim=8, activation='relu'))
nn.add(Dense(8, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))
# compile the keras model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
nn.fit(X, y, epochs=150, batch_size=10,verbose=0)
# evaluate the keras model
_, accuracy = nn.evaluate(X, y,verbose=0)
print('Accuracy: %.2f' % (accuracy*100))
# predict classes using your model
predictions = nn.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

# Path: blockchainlab\sklearn.py
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# load your data
mydata = loadtxt('export-SGO.csv', delimiter=',')
# partition data into input (X) and output (y)
X = mydata[:,0:8]
y = mydata[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
# define the model
model = LogisticRegression()
# fit the model on the training set
model.fit(X_train, y_train)
# make predictions on the test set
predictions = model.predict(X_test)
# evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.2f' % (accuracy*100))
