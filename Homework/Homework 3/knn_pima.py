"""
@author: ari
"""
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer)

'''
Train a k-nearest neighbor classifier to the Pima Indians data. Split the data 50/50 into training and test sets using
train_test_split() with random_state=1 to provide a fixed seed. Try out different approaches for feature scaling and
selection to see their effects on the test set classification accuracy. Also experiment with different values for the
number of neighbors 𝑘. Report the highest test set accuracy you are able to obtain.
'''

# Load the data
pima = read_csv('pima-indians-diabetes.csv', header=None)
X = pima.iloc[:, :-1].values
y = pima.iloc[:, -1].values

# Split the data 50/50 into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


def standardize_and_test(scalar_type, scalar):
    """
    Standardizes the data and tests the accuracy of the model Parameters ---------- scalar_type : str - The type of
    scaling being used scalar : StandardScaler, MinMaxScaler, None, MaxAbsScaler, RobustScaler, Normalizer,
    QuantileTransformer, PowerTransformer

    Returns the best accuracy and k for the given scalar
    -------

    """
    # Standardize the data
    if scalar is not None:
        scalar_x_train = scalar.fit_transform(X_train)
        scalar_x_test = scalar.transform(X_test)
    else:
        scalar_x_train = X_train.copy()
        scalar_x_test = X_test.copy()

    # Best accuracy and k
    highest_accuracy, k_pos = 0, 0

    print(scalar_type + ':')
    # Try out different values for the number of neighbors 𝑘
    for ki in range(1, 10):
        # Train a k-nearest neighbor classifier
        knn = KNeighborsClassifier(n_neighbors=ki)
        knn.fit(scalar_x_train, y_train)

        # Test the accuracy of the model
        score = knn.score(scalar_x_test, y_test)

        # If the current accuracy is better than the current best accuracy, update the best accuracy and ki
        if score > highest_accuracy:
            highest_accuracy = score
            k_pos = ki

        # Report the highest test set accuracy you are able to obtain
        print('k =', ki, 'Test set accuracy: {:0.2f}%'.format(score * 100.0))
    print()
    return highest_accuracy, k_pos


# Scalars/Labels to test
scalars = [StandardScaler(), MinMaxScaler(), None, MaxAbsScaler(), RobustScaler(), Normalizer(), QuantileTransformer(),
           PowerTransformer()]
labels = ['Standard Scalar', 'MinMax Scalar', 'No Scalar', 'MaxAbs Scalar', 'Robust Scalar', 'Normalizer',
          'Quantile Transformer', 'Power Transformer']

# Best accuracy, k, and scalar
best_accuracy, best_k, best_scalar = 0, 0, None

# Iterate through the scalars to find the best accuracy
for i in range(len(scalars)):
    # Get the best accuracy and k for the current scalar
    accuracy, k = standardize_and_test(labels[i], scalars[i])

    # If the current accuracy is better than the current best accuracy, update the best accuracy, k, and scalar
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        best_scalar = labels[i]

# Report the highest test set accuracy obtained
print("Best Scalar: {:0.2f}%".format(best_accuracy * 100.0))
print('Best k:', best_k)
print('Best Scalar:', best_scalar)
