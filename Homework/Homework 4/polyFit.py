"""
Train a polynomial regression on the data in trainPoly.csv using ordinary least squares (OLS)
for each value of maximum degree 𝑑 = 1, …, 9. Do not first standardize or use any other feature
scaling! Use the trained model to make and evaluate predictions on the data in testPoly.csv.
In both files, the first column contains the input, and the second column contains the prediction
target. Submit your code in a file named polyFit.py.
a) Plot the mean squared error (MSE) for the training data and the test data on the same axes
as a function of maximum degree 𝑑. Include axis labels and a legend in your plot. Explain
the trend in the plot.
b) Plot the normalized squared magnitude of the weight vector ‖𝒘‖2/𝑑 on a log scale as
function of 𝑑. Include axis labels and a legend in your plot. Explain the trend in the plot.
c) Create the same two plots using ridge regression (L2 penalty) with regularization strength
𝛼 = 10−6 instead of OLS and compare the results with OLS.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Load train data
train = np.loadtxt('trainPoly.csv', delimiter=',')
x_train = train[:, 0]
y_train = train[:, 1]

# Load test data
test = np.loadtxt('testPoly.csv', delimiter=',')
x_test = test[:, 0]
y_test = test[:, 1]

# Reshape data
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

degree = range(1, 10)

# MSE data
mse_train = []
mse_test = []

# Weight vector data
w_norm = []

# Ridge regression data
mse_train_ridge = []
mse_test_ridge = []

# Ridge regression weight vector data
w_norm_ridge = []

# Create polynomial features
for d in degree:
    poly = PolynomialFeatures(degree=d)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    # Train model
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # Predict
    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    # Calculate MSE
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

    # Calculate weight vector
    w_norm.append(np.linalg.norm(model.coef_) / d)

    # Ridge regression
    ridge = Ridge(alpha=10**-6)
    ridge.fit(x_train_poly, y_train)

    # Predict
    y_train_pred_ridge = ridge.predict(x_train_poly)
    y_test_pred_ridge = ridge.predict(x_test_poly)

    # Calculate MSE
    mse_train_ridge.append(mean_squared_error(y_train, y_train_pred_ridge))
    mse_test_ridge.append(mean_squared_error(y_test, y_test_pred_ridge))

    # Calculate weight vector
    w_norm_ridge.append(np.linalg.norm(ridge.coef_) / d)

# Plot MSE
plt.plot(degree, mse_train, 'bo')
plt.plot(degree, mse_test, 'ro')
plt.xlabel('Degree')
plt.ylabel('MSE')
plt.legend(['Train', 'Test'])
plt.title('MSE for train and test data')
plt.show()

# Plot Weight Vector Plot
plt.plot(degree, w_norm, 'bo')
plt.xlabel('Degree')
plt.ylabel('Weight vector')
plt.title('Weight vector for train data')
plt.show()

# Plot MSE Ridge
plt.plot(degree, mse_train_ridge, 'bo')
plt.plot(degree, mse_test_ridge, 'ro')
plt.xlabel('Degree')
plt.ylabel('MSE')
plt.legend(['Train', 'Test'])
plt.title('MSE for train and test data with Ridge')
plt.show()

# Plot Weight Vector Plot Ridge
plt.plot(degree, w_norm_ridge, 'bo')
plt.xlabel('Degree')
plt.ylabel('Weight vector')
plt.title('Weight vector for train data with Ridge')
plt.show()

"""
Writeup:
a) Plot the mean squared error (MSE) for the training data and the test data on the same axes
as a function of maximum degree 𝑑. Include axis labels and a legend in your plot. Explain
the trend in the plot.

The MSE for the training data decreases as the degree increases. This is because the model is
overfitting the training data. The MSE for the test data increases as the degree increases. This
is because the model is overfitting the training data and not generalizing well to the test data.


b) Plot the normalized squared magnitude of the weight vector ‖𝒘‖2/𝑑 on a log scale as
function of 𝑑. Include axis labels and a legend in your plot. Explain the trend in the plot.

The weight vector increases as the degree increases. This is because the model is overfitting
the training data and the weight vector is increasing to fit the training data.


c) Create the same two plots using ridge regression (L2 penalty) with regularization strength
𝛼 = 10−6 instead of OLS and compare the results with OLS.

The MSE for the training data decreases as the degree increases. This is because the model is
overfitting the training data. The MSE for the test data increases as the degree increases. This
is because the model is overfitting the training data and not generalizing well to the test data.
"""