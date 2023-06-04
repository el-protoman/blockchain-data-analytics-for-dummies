from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()
plt.figure(dpi = 300)
clf = tree.DecisionTreeClassifier(random_state=0).fit(iris.data, iris.target)

tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.tight_layout()
plt.savefig('decisionTree.png', dpi=300)
plt.show()