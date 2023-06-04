import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load your data
df = pd.read_csv('export-SGO.csv', delimiter=',', skiprows=1)
df_cleaned = df.apply(pd.to_numeric, errors='coerce')
data_types = df_cleaned.dtypes
print(data_types)
print(df_cleaned.head())
#preprocess the data
mydata = df_cleaned

#check for missing values
# if np.isnan(mydata).any():
#     # Handle missing values by removing rows or imputing
#     mydata = mydata[~np.isnan(mydata).any(axis=1)]

#partition data into input (X) and output (y)
X = mydata.iloc[:,:-1].values
print(X)
y = mydata.iloc[:,-1].values

#convert data types
X = X.astype(np.float32)
y = y.astype(np.int64)

# Check if the data contains valid samples
if X.shape[0] == 0 or y.shape[0] == 0:
    print("No valid samples in the data.")
    # Handle this case appropriately
    # ...
else:
    # Check unique values in y_train
    unique_values = np.unique(y)
    if len(unique_values) < 2:
        print("Target values do not have enough unique classes.")
        # Handle this case appropriately
        # ...
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
    # Convert numpy array to torch tensor
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    # Set hyperparameters and continue with the rest of your code
    # ...

#set hyperparameters
#inputUnits = 8
inputUnits = X_train.shape[1]
hiddenUnits = 12
outputUnits = 1
learningRate = 0.01
epochs = 150

#define linear regression model
regressModel = torch.nn.Sequential(
    torch.nn.Linear(inputUnits, hiddenUnits),
    torch.nn.ReLU(),
    torch.nn.Linear(hiddenUnits, outputUnits),
)
#define loss function
lossFunction = torch.nn.CrossEntropyLoss()

#define optimization algorithm
optimizer = torch.optim.Adam(regressModel.parameters(), lr=learningRate)

#train the model
for epoch in range(epochs):
    #convert numpy array to torch tensor
    # inputs = torch.from_numpy(X_train).float()
    # targets = torch.from_numpy(y_train).long()
    #forward pass
    #outputs = regressModel(inputs)
    outputs = regressModel(X_train)
    #loss = lossFunction(outputs, targets)
    loss = lossFunction(outputs, y_train)
    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print loss
    print('epoch {}, loss {}'.format(epoch, loss.item()))

#test the model
#inputs = torch.from_numpy(X_test).float()
#targets = torch.from_numpy(y_test).long()
#outputs = regressModel(inputs)
outputs = regressModel(X_test)
_, predicted = torch.max(outputs, 1)
#print('Accuracy: %.2f' % (accuracy_score(targets, predicted)*100))
print('Accuracy: %.2f' % (accuracy_score(y_test, predicted)*100))
# summarize the first 5 cases
for i in range(5):
    # print('%s => %d (expected %d)' % (X_test[i].tolist(), predicted[i], targets[i]))
    print('%s => %d (expected %d)' % (X_test[i].tolist(), predicted[i], targets[y_test[i]]))

# #save the model
# torch.save(regressModel.state_dict(), 'regressModel.pt')
# #load the model
# regressModel = torch.nn.Sequential(
#     torch.nn.Linear(inputUnits, hiddenUnits),
#     torch.nn.ReLU(),
#     torch.nn.Linear(hiddenUnits, outputUnits),
# )
# regressModel.load_state_dict(torch.load('regressModel.pt'))
# regressModel.eval()
# #predict classes using your model
# inputs = torch.from_numpy(X_test).float()
# outputs = regressModel(inputs)
# _, predicted = torch.max(outputs, 1)
# # summarize the first 5 cases
# for i in range(5):
#     print('%s => %d (expected %d)' % (X_test[i].tolist(), predicted[i], targets[i]))

# Path: blockchainlab\pytorch.py
# from numpy import loadtxt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# # load your data
# mydata = loadtxt('export-SGO.csv', delimiter=',')
# # partition data into input (X) and output (y)
# X = mydata[:,0:8]
# y = mydata[:,8]
# # split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
# # define the model
# model = LogisticRegression()
# # fit the model on the training set
# model.fit(X_train, y_train)
# # make predictions on the test set
# predictions = model.predict(X_test)
# # evaluate accuracy
# accuracy = accuracy_score(y_test, predictions)
# print('Accuracy: %.2f' % (accuracy*100))
# # summarize the first 5 cases
# for i in range(5):
#     print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test[i]))
#