import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

dfpreprocess = pd.read_csv('products.csv',usecols=['screenType','resolution'])
screenTypeList = dfpreprocess.screenType.unique().tolist()
resolutionList = dfpreprocess.resolution.unique().tolist()

print(screenTypeList)
print(resolutionList)

fscreen = lambda x: screenTypeList.index(x)
fresolution = lambda x: resolutionList.index(x)

converter = {'screenType': fscreen, 'resolution': fresolution}

print(converter)

df = pd.read_csv('products.csv',usecols=['screenSize','screenType','resolution','hdmiPorts','usbPorts','reviewRating'],converters=converter)

# By using the converters parameter in the pd.read_csv() function, you can specify custom conversion functions to transform specific columns from text to numeric values during the data loading process
# The converters parameter is a dictionary where the key is the column name and the value is the conversion function
# The conversion function takes a single parameter, the value of the column, and returns the converted value
# The conversion function can be a lambda function or a regular function
# The conversion function can be defined in the same line as the converters parameter or it can be defined as a separate function
# 
# The following code snippet shows how to use a lambda function to convert the screenType column from text to numeric values:
# fscreen = lambda x: screenTypeList.index(x)
# 
# The following code snippet shows how to use a regular function to convert the resolution column from text to numeric values:
# def fresolution(x):
#    return resolutionList.index(x)
#  

y = df.reviewRating
X = df.drop('reviewRating',axis=1)

# the test_size parameter specifies the percentage of the data that should be used for testing (25% in this case)
# the rest of the data is used for training (75% in this case)
# the random_state parameter is used to seed the random number generator so that the same random numbers are generated each time the code is run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#feature_names = X.columns.tolist()  # Get the column names of X
feature_names = ['screenSize', 'screenType', 'resolution', 'hdmiPorts', 'usbPorts']  # Exclude 'reviewRating' from feature_names

plt.figure(dpi = 300)
clf = tree.DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

# Set the feature names explicitly for the DecisionTreeClassifier
clf.feature_names = feature_names
tree.plot_tree(clf, filled=True, class_names=['1', '2', '3', '4', '5'])

#tree.plot_tree(clf, feature_names=feature_names, filled=True, class_names=['1', '2', '3', '4', '5'])
# class_names parameter is used to specify the class names for the target variable (reviewRating in this case) in the order of the numeric values (0, 1, 2, 3, 4, 5)

plt.tight_layout()
plt.savefig('decisionTree.png', dpi=300)
plt.show()

# The following code snippet shows how to use the predict() function to predict the reviewRating for a new product:
# the predict() function takes a list of lists as a parameter where each inner list contains the values for the features of the new product
# the predict() function returns a list of the predicted reviewRating values for each of the new products
# the following code snippet shows how to predict the reviewRating for a new product with the following features: screenSize = 15.6, screenType = 0, resolution = 0, hdmiPorts = 1, usbPorts = 2
predicted_ratings = clf.predict([[15.6, 0, 0, 1, 2]])
print(predicted_ratings)

# The following code snippet shows how to use the score() function to calculate the accuracy of the model:
score = clf.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score))