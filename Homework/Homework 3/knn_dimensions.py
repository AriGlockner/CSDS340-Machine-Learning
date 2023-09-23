import numpy as np


def generate_data(d):
    """
    Generate 1,000 random data points in d dimensions, where each dimension is uniformly distributed between -1 and 1

    :param d: number of dimensions
    :return: 1,000 random data points in d dimensions
    """
    return np.random.uniform(-1, 1, (1000, d))


# Plot the following measures as you increase the number of dimensions d from 2 to 10:
np.random.seed(1)

# Get the data
data = []
for i in range(2, 11):
    data.append(np.random.uniform(-1, 1, (1000, i)))
print(data)

'''
a)
The fraction of data points within the unit hypersphere, i.e. the fraction of data points with
distance ≤ 1 from the origin (all zero coordinates). This measures roughly what fraction
of data points are close to a typical (all zero) data point
'''


'''
b)
The mean distance between a data point and its 1-nearest neighbor divided by the mean
distance between any pair of data points. This measures how close a nearest neighbor is
relative to a randomly selected data point. (As the number of dimensions increases, the
mean distance between any pair of data points also increases, so we divide by this to pro-
vide a fair comparison.)
'''
