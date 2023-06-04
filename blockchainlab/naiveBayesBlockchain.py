import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
dfpreprocess = pd.read_csv('products.csv', usecols=['screenType', 'resolution'])
screenTypeList = dfpreprocess.screenType.unique().tolist()
resolutionList = dfpreprocess.resolution.unique().tolist()

print(screenTypeList)
print(resolutionList)

fscreen = lambda x: screenTypeList.index(x)
fresolution = lambda x: resolutionList.index(x)

converter = {'screenType': fscreen, 'resolution': fresolution}

print(converter)

df = pd.read_csv('products.csv', usecols=['screenSize', 'screenType', 'resolution', 'hdmiPorts', 'usbPorts', 'reviewRating'], converters=converter)

# Split data into training and testing sets
y = df.reviewRating
X = df.drop('reviewRating', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Define feature names
feature_names = ['screenSize', 'screenType', 'resolution', 'hdmiPorts', 'usbPorts']  # Exclude 'reviewRating' from feature_names

# Train the Naive Bayes classifier
nbClass = GaussianNB()
nbClass.fit(X_train, y_train)

# Predict on the test set
y_predict = nbClass.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy: {:.2f}".format(accuracy))

# Plotting code if needed
plt.figure(dpi=300)
# Count the number of occurrences for each review rating
rating_counts = df['reviewRating'].value_counts().sort_index()

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(rating_counts.index, rating_counts.values)
plt.xlabel('Review Rating')
plt.ylabel('Count')
plt.title('Distribution of Review Ratings')
plt.show()

# Generate random data following a Gaussian distribution
mu = 0      # mean
sigma = 1   # standard deviation
data = np.random.normal(mu, sigma, 1000)

# Plot the histogram of the data
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')

# Plot the Gaussian distribution curve
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
plt.plot(x, y, color='orange', linewidth=2)

# Set plot labels and title
plt.xlabel('Data')
plt.ylabel('Probability Density')
plt.title('Gaussian Distribution')

# Display the plot
plt.show()