import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


def plot_graph(ydata, y_label):
    plt.plot(range(2, 11), ydata)
    plt.xlabel('Dimensions')
    plt.ylabel(y_label)
    plt.show()
    pass


# Set the random seed, so I can use the same random set of data in my tests
np.random.seed(1)

# Part A: Calculate the percent of points within the unit hypersphere
percent = []

# Part B: Calculate the mean distance between a data point and its 1-nearest neighbor
mean_distance = []

# For each dimension from 2 to 10
for dimension in range(2, 11):
    # Generate data for each dimension
    data = np.random.uniform(-1, 1, (1000, dimension))

    # Part A - Calculate the number of points within the unit hypersphere
    percent.append(sum(1 for point in data if np.linalg.norm(point) <= 1) / len(data))

    # Part B - Calculate the mean distance between a data point and its 1-nearest neighbor
    distances, indices = (NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)
                          .kneighbors(data))
    mean_distance.append(np.mean(distances[:, 1]))

# Plot part A
plot_graph(percent, 'Percent of Points within the Unit Hypersphere')
# Plot part B
plot_graph(mean_distance, 'Mean Distance between a Data Point and its 1-Nearest Neighbor')
