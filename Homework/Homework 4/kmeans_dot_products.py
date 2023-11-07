from sklearn.cluster import KMeans
import numpy as np


def api_fit_kmeans_dot_products(K, n_clusters, max_iter=300):
    """
    Use the api to fit a K-means model using dot products as the distance metric.

    Parameters
    ----------
    K : array-like, shape (n_samples, n_features)
        Training instances to cluster.
    n_clusters : int
        Number of clusters to form.
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm for a
        single run.

    Returns SSE
    -------

    """
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init="auto")
    kmeans.fit(K)
    return kmeans.inertia_


def fit_kmeans_dot_products(K, n_clusters, max_iter=300):
    """
    Fit a K-means model using dot products as the distance metric manually.

    Parameters
    ----------
    K : array-like, shape (n_samples, n_features)
        Training instances to cluster.
    n_clusters : int
        Number of clusters to form.
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm for a
        single run.

    Returns SSE
    -------

    """

    # Convert input data to numpy array
    K = np.array(K)

    # Initialize centroids as the first n_clusters points in the data
    centroids = K[:n_clusters]

    for iteration in range(max_iter):
        # Assign points to clusters
        clusters = [[] for _ in range(n_clusters)]
        for point in K:
            cluster = get_closest_centroid(point, centroids)
            clusters[cluster].append(point)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                # Calculate mean for each dimension separately
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)

        # Check if centroids have changed
        new_centroids = np.array(new_centroids)
        if np.array_equal(new_centroids, centroids):
            return round(sum([get_sse(cluster, centroid) for cluster, centroid in zip(clusters, centroids)]), 5)
        else:
            centroids = new_centroids

    return -1


def get_sse(cluster, centroid):
    """
    Get the sum of squared errors for a cluster.
    Parameters
    ----------
    cluster
    centroid

    Returns
    -------

    """
    return sum([get_distance(point, centroid) ** 2 for point in cluster])


def get_closest_centroid(point, clusters):
    """
    Get the nearest centroid of a point.
    Parameters
    ----------
    point
    clusters

    Returns
    -------

    """
    distances = [get_distance(point, centroid) for centroid in clusters]
    return distances.index(min(distances))


def get_distance(point1, point2):
    """
    Get the distance between two points.
    Parameters
    ----------
    point1
    point2

    Returns
    -------

    """
    return sum([(x - y) ** 2 for x, y in zip(point1, point2)]) ** 0.5


if __name__ == "__main__":
    input_K = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    n = 2

    print("Baseline:", api_fit_kmeans_dot_products(input_K, n))
    print("My Implementation:", fit_kmeans_dot_products(input_K, n))
