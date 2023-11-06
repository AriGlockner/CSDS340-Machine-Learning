import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering


# Load the 2 moons data
data = np.loadtxt('twomoons.csv', delimiter=',')


def plot_settings(title, c=data[:, 2]):
    plt.scatter(data[:, 0], data[:, 1], c=c, cmap='bwr')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()


# Using only the 1st 2 features, recover the true clusters using K-means, Agglomerative Hierarchical, and Spectral
# clustering
labels = ["K-means", "Agglomerative Hierarchical", "Spectral"]
models = [KMeans(n_clusters=2, random_state=1), AgglomerativeClustering(n_clusters=2),
          SpectralClustering(n_clusters=2, random_state=1)]

for i, model in enumerate(models):
    # Fit model
    model.fit(data[:, 0:2])

    # Predict
    y_pred = model.labels_

    # Plot
    plot_settings(labels[i], y_pred)

# Actual
plot_settings("Actual")

"""
Default Hyperparameters (if applicable):
Randomized_State = 1
Number of Cluster = 2

Results:
While none of the 3 models were 100% correct, the Agglomerate Hierarchical model was the least accurate. The
Agglomerate was completely correct with one of the classes, but was very far off with the other class.

The K-means and the Spectral Clusters produced the exact same results.
"""