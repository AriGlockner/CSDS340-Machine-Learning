import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class PerceptronModel:
    """
    Perceptron classifier.

    Parameters
    ----------
    eta : float - Learning rate (between 0.0 and 1.0)
    n_iterations : int - Passes over the training dataset.
    random_state : int - Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array - Weights after fitting.
    b_ : Scalar - Bias unit after fitting.
    errors_ : list - Number of misclassifications (updates) in each epoch.
    predictions_ : list - Predictions for each epoch.

    """

    # Weights - to be updated during training
    w_ = None
    # Bias Units - to be updated during training
    b_ = None
    # Number of updates in each epoch
    errors_ = None
    # Predictions for each epoch
    predictions_ = None

    def __init__(self, eta=0.0001, n_iterations=50, random_state=1):
        """
        :param eta: Learning rate (between 0.0 and 1.0)
        :param n_iterations: Passes over the training dataset.
        :param random_state: Random number generator seed for random weight initialization
        """

        self.eta = eta
        self.n_iterations = n_iterations
        self.random_state = random_state
        pass

    def fit(self, X, y):
        """
        Fit training data.
        :param X: Training vectors, where n_examples is the number of examples and n_features is the number of features.
        :param y: Target values.
        :return: self
        """

        random_generation = np.random.RandomState(self.random_state)
        self.w_ = random_generation.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)

        self.errors_ = []
        self.predictions_ = []

        for i in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))

                for j in range(len(self.w_)):
                    self.w_[j] += update * xi[j]
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            self.predictions_.append(self.predict(X))
        return self

    def net_input(self, X):
        """
        Calculate net input
        :param X: Training vectors, where n_examples is the number of examples and n_features is the number of features.
        :return: net input
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        This is the part that makes this a Perceptron
        :param X:
        :return: Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.5, 1, 0)


# Read the dataset into a pandas.DataFrame - This link pulls up the same data as the csv file
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")

# Split the dataset into features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Standardize the data
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

'''
Part a) Find the learning rate with the highest accuracy
'''

print('Part a) Testing the learning rates')
learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

for rate in learning_rates:
    perceptron_learn_rate = PerceptronModel(eta=rate)
    perceptron_learn_rate.fit(X_train, y_train)
    print("Learning Rate: ", rate, ", Accuracy: ", accuracy_score(y_test, perceptron_learn_rate.predict(X_test)))

'''
Part b) Plot the number of updates for each epoch for with the highest accuracy learning rate
'''

# Test the perceptron model with the highest accuracy learning rate
perceptron = PerceptronModel(eta=0.0001, n_iterations=60, random_state=1)
perceptron.fit(X_train, y_train)

# Plot the results of the perceptron model with the highest accuracy learning rate
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, 'bo-', label='Training updates')
plt.xlabel('Epochs')
plt.ylabel('Number of Updates')
plt.title('Part b')
plt.show()
