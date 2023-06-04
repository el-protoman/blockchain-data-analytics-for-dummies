from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
class_names = iris.target_names
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], iris_df['target'], test_size=0.3, random_state=0)
nbClass = GaussianNB()
nbClass.fit(X_train, y_train)
y_predict = nbClass.predict(X_test)

print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_predict)))
