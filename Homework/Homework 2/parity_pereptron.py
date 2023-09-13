import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PerceptronModel:
    # Weights - to be updated during training
    w_ = None
    # Bias Units - to be updated during training
    b_ = None
    # Number of updates in each epoch
    errors_ = None
    # Predictions for each epoch
    predictions_ = None

    def __init__(self, eta=0.1, n_iterations=100):
        """
        Parameters
        ----------
        eta : float - Learning rate (between 0.0 and 1.0)
        """

        self.n_iterations = n_iterations
        self.eta = eta

    def fit(self, X, y, w, b):
        # Initialize weights and bias units to the ones passed in
        self.w_ = w
        self.b_ = b

        self.errors_ = []
        self.predictions_ = []

        for _ in range(self.n_iterations):
            errors = 0
            prediction = 0

            # Train the perceptron
            for xi, target in zip(X, y):
                # Calculate the prediction/update
                prediction += self.predict(xi)
                update = self.eta * (target - prediction)

                # Update the weights and bias units
                self.w_ += update * xi
                self.b_ += update

                # Update the errors
                errors += np.where(update[update != 0.0], 1, 0).size

            # Update the prediction/errors for this epoch
            self.errors_.append(errors)
            self.predictions_.append(prediction)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 0, 1)

    def test(self, X):
        predictions = []

        for xi, target in zip(X, y):
            predict = min(self.predict(xi).sum(), 1)
            # print(predict)
            predictions.append(predict)
        # print(predictions)
        return predictions


'''
The parity problem returns 1 if the number of inputs that are 1 is even, and 0 otherwise. Can a
Perceptron learn this problem for 3 inputs? Design the network and train it to the entire data set.
Submit the table of input and target combinations, a diagram of the network structure, and the
highest training set accuracy you are able to obtain with a Perceptron along with the weights that
achieve this accuracy.
'''

# All possible inputs/outputs
X = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
y = [1, 0, 0, 1, 0, 1, 1, 0]

# Create a Perceptron
# perceptron = PerceptronModel()

# Split the dataset into training and test sets
X_train = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]]
X_test = [[0, 1, 0], [1, 0, 0], [1, 0, 1]]
y_train = [1, 0, 1, 1, 0]
y_test = [0, 1, 0]

'''
Test for the optimal learning rate
'''
best_accuracy = 0
best_w = 0.0
best_b = 0.0
best_eta = 0.0

w = 0.1
b = 0.1
eta=0.1
for _ in range(9):
    for __ in range(9):
        perceptron = PerceptronModel(eta)

        perceptron.fit(X_train, y_train, w, b)
        accuracy = accuracy_score(y_test, perceptron.test(X_test))
        print(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_w = w
            best_b = b
            best_eta = eta

        b += 0.1
    w += 0.1

print('Best Accuracy: ', best_accuracy)
print('Best Weights: ', best_w)
print('Best Bias: ', best_b)
print('Best Learning Rate: ', best_eta)

'''
perceptron.fit(X_train, y_train, 0.1, 0.1)

accuracy = accuracy_score(y_test, perceptron.test(X_test))
print('Accuracy: ', accuracy)
'''
