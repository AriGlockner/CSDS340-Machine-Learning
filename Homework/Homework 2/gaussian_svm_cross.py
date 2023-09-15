import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


def plot_data(d, classifier, title):
    # Plot the data
    for xi, yi in zip(d[0], d[1]):
        if yi == 0:
            plt.scatter(xi[0], xi[1], c='red', marker='o', label='1')
        else:
            plt.scatter(xi[0], xi[1], c='blue', marker='x', label='0')

    # Plot settings
    plt.title(title)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])

    # Plot the decision regions
    X = np.array(d[0])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=ListedColormap(('red', 'blue')))
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.show()


# Data
xs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
os = [(-2, 0), (0, 2), (2, 0), (0, -2)]
X = xs + os
y = [1, 1, 1, 1, 0, 0, 0, 0]

# Kernal types
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernal in kernels:
    # Fit SVM model
    svm = SVC(kernel=kernal, C=2.0, random_state=1)
    svm.fit(X, y)

    # Plot the decision regions
    plot_data([X, y], svm, kernal + " SVM")

'''
a) What is the maximum training set accuracy achievable by a linear SVM?
50%. The data is not linearly separable, so the maximum accuracy is 50%.

b) Find a kernal function for an SVM that perfectly classifies the data.
The rbf kernel function perfectly classifies the data.

c) Can a Gaussian kernal SVM perfectly separate the two classes? If so, fit one to the training data and plot its
decision region. Submit your plotted decision region.
Yes, the Gaussian kernel SVM can perfectly separate the two classes.
'''

# Show the plotted decision region for the ideal SVM
svm = SVC(kernel='rbf', C=2.0, random_state=1)
svm.fit(X, y)
plot_data([X, y], svm, "Ideal SVM")
