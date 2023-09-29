import numpy as np
from matplotlib import pyplot as plt

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]
ticks = [0, 0.5, 1, 1.5]


def plot_data(show=True):
    for i in range(4):
        if y[i] == 0:
            plt.plot(X[i][0], X[i][1], 'ro')
        else:
            plt.plot(X[i][0], X[i][1], 'bo')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(-0.1, 1.5)
    plt.ylim(-0.1, 1.5)
    if show:
        plt.show()


# Hard coded Perceptron
t = np.linspace(0, 2, 10)
plt.plot(t, 1 - t, 'c')
plt.title('Trained Perceptron Boundary')
plot_data()

# Hard coded Trained Decision boundary
t = np.linspace(0.5, 2, 10)
constant = np.linspace(0.5, 0.5, 10)
plt.plot(constant, t, 'c')
plt.plot(t, constant, 'c')
plt.title('Trained Decision Tree Boundary')
plot_data()

# Data points for logical AND function
data = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

# Final weights and bias obtained from training
w1, w2, b = 1, 1, -1

# Function to calculate decision boundary
def decision_boundary(x):
    return (-w1 * x - b) / w2

# Plotting the data points
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data Points')

# Plotting the decision boundary
x_values = np.linspace(-0.5, 1.5, 400)  # Generate 400 points between -0.5 and 1.5 for x-axis
plt.plot(x_values, decision_boundary(x_values), color='red', label='Decision Boundary')

plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Perceptron Decision Boundary for Logical AND')
plt.legend()
plt.grid(True)
plt.show()