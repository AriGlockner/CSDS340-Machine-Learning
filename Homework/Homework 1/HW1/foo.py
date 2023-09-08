import numpy as np
import matplotlib.pyplot as plt


# Define the perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, num_epochs=100):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros(input_size)
        self.bias = 0

    def activation_function(self, x):
        # Step function: 1 if x >= 0, else 0
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.activation_function(z)

    def train(self, X, y):
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error


# Generate a simple dataset for training
X = np.array([[2, 3], [3, 3], [1, 2], [5, 1]])
y = np.array([1, 1, 0, 0])

# Create a perceptron instance
perceptron = Perceptron(input_size=2)

# Train the perceptron
perceptron.train(X, y)

# Visualize the trained decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Trained Perceptron Decision Boundary")
w, b = perceptron.weights, perceptron.bias
x_boundary = np.linspace(0, 5, 100)
y_boundary = (-w[0] / w[1]) * x_boundary - (b / w[1])
plt.plot(x_boundary, y_boundary, '-r', label='Decision Boundary')
plt.legend()
plt.show()

# Test the trained perceptron
test_inputs = np.array([[4, 2], [1, 3]])
for x in test_inputs:
    prediction = perceptron.predict(x)
    print(f"Input: {x}, Prediction: {prediction}")
