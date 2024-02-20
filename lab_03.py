#Load IRIS dataset, check its contents:

from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

print("Feature Names:")
print(iris.feature_names)

print("\nFirst 5 rows of data:")
print(iris.data[0:5,:])

print("\nTarget Values for the First 5 Samples:")
print(iris.target[0:5])


#Split data into training and testing parts:

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


#Use a Support Vector Machine for classification:

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

SVMmodel = SVC(kernel='linear')
SVMmodel.fit(X_train, y_train)

print("Parameters of the SVM model:")
print(SVMmodel.get_params())

# Evaluate the accuracy of the model on the testing set
accuracy = SVMmodel.score(X_test, y_test)
print("Accuracy of the SVM model on the testing set:", accuracy)

#Let's explore more now.

#Choose only first two features (columns) of iris.data
#SVM is in its basic form a 2-class classifier, so eliminate iris.target =2 from the data

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, :2]

y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

SVMmodel = SVC(kernel='linear')
SVMmodel.fit(X_train, y_train)

accuracy = SVMmodel.score(X_test, y_test)
print("Accuracy of the SVM model on the testing set:", accuracy)

#Plot scatterplots of targets 0 and 1 and check the separability of the classes:

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, :2]

y = iris.target[(iris.target == 0) | (iris.target == 1)]

X_selected = X[(iris.target == 0) | (iris.target == 1)]

plt.figure(figsize=(8, 6))
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Scatter plot of Iris dataset (classes 0 and 1)')
plt.colorbar(label='Classes', ticks=[0, 1])
plt.grid(True)
plt.show()

#Train and test the SVM classifier, play with regularization parameter C (either use the default value or try e.g. 200)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, :2]

y = iris.target[(iris.target == 0) | (iris.target == 1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

SVMmodel_default_C = SVC(kernel='linear')
SVMmodel_default_C.fit(X_train, y_train)

accuracy_default_C = SVMmodel_default_C.score(X_test, y_test)
print("Accuracy with default C value:", accuracy_default_C)

SVMmodel_custom_C = SVC(kernel='linear', C=200)
SVMmodel_custom_C.fit(X_train, y_train)

accuracy_custom_C = SVMmodel_custom_C.score(X_test, y_test)
print("Accuracy with custom C value (C=200):", accuracy_custom_C)


#Show support vectors in the 2D plot, plot the decision line from equation [w0 w1]*[x0 x1] + b = 0:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, :2]

y = iris.target[(iris.target == 0) | (iris.target == 1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

SVMmodel = SVC(kernel='linear')
SVMmodel.fit(X_train, y_train)

support_vectors = SVMmodel.support_vectors_
coef = SVMmodel.coef_[0]
intercept = SVMmodel.intercept_

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', label='Training data')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red', marker='o', s=100, label='Support Vectors')

x0 = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
x1 = -(coef[0] / coef[1]) * x0 - intercept / coef[1]
plt.plot(x0, x1, color='blue', linestyle='-', label='Decision Line')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundary with Support Vectors')
plt.legend()
plt.grid(True)
plt.show()

#Import one-class SVM and generate data (Gaussian blobs in 2D-plane):

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.3, random_state=42)

outliers = np.random.uniform(low=-3, high=3, size=(20, 2))

X = np.vstack([X, outliers])

model = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
model.fit(X)

y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Paired, edgecolors='k')
plt.title('One-Class SVM for Outlier Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.colorbar(label='Predicted Class')
plt.show()


#Train one-class SVM and plot the outliers (outputs of prediction being equal to -1)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.3, random_state=42)

outliers = np.random.uniform(low=-3, high=3, size=(20, 2))

X = np.vstack([X, outliers])

model = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
model.fit(X)

y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Inliers')
plt.scatter(X[y_pred == -1][:, 0], X[y_pred == -1][:, 1], c='red', label='Outliers')
plt.title('One-Class SVM for Outlier Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.legend()
plt.show()


#Plot the support vectors:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.3, random_state=42)

outliers = np.random.uniform(low=-3, high=3, size=(20, 2))

X = np.vstack([X, outliers])

model = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
model.fit(X)

y_pred = model.predict(X)

support_vectors = model.support_vectors_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Inliers')
plt.scatter(X[y_pred == -1][:, 0], X[y_pred == -1][:, 1], c='red', label='Outliers')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='green', marker='o', s=100, label='Support Vectors')
plt.title('One-Class SVM for Outlier Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.legend()
plt.show()


#What if we want to have a control what is outlier? Use e.g. 5% "quantile" to mark the outliers. Every point with lower score than threshold will be an outlier.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.3, random_state=42)

outliers = np.random.uniform(low=-3, high=3, size=(20, 2))

X = np.vstack([X, outliers])

model = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
model.fit(X)

scores = model.score_samples(X)

threshold = np.quantile(scores, 0.05)

outliers_index = np.where(scores <= threshold)[0]
outliers_values = X[outliers_index]

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], label='Inliers')
plt.scatter(outliers_values[:, 0], outliers_values[:, 1], color='red', label='Outliers')
plt.title('One-Class SVM for Outlier Detection (Custom Threshold)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
