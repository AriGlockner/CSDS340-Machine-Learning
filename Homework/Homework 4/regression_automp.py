import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


if __name__ == '__main__':
    # Load Auto-Mpg data
    auto_mpg = np.loadtxt('auto-mpg-missing-data-removed.txt', comments='"')

    # Reshape data
    auto_mpg = auto_mpg.reshape(-1, 8)
    scaler = StandardScaler()
    scaler.fit(auto_mpg[:, 1:8])
    auto_mpg[:, 1:8] = scaler.transform(auto_mpg[:, 1:8])

    # Split data using train_test_split
    train, test = train_test_split(auto_mpg, test_size=0.5, random_state=1)

    # Types of Regression
    labels = ["Elastic net penalized linear regression with polynomial features", "Support Vector Regression (linear)",
              "Support Vector Regression (polynomial)", "Support Vector Regression (RBF)", "Random Forest Regression",
              "K-Nearest Neighbors Regression (KNN)"]

    # Regression models
    models = [ElasticNet(), SVR(kernel='linear', max_iter=10000), SVR(kernel='poly', max_iter=10000),
              SVR(kernel='rbf', max_iter=10000), RandomForestRegressor(n_estimators=1000), KNeighborsRegressor()]

    # Report the maximum test set R^2 value for each model
    for i, model in enumerate(models):
        # Create polynomial features
        poly = PolynomialFeatures(degree=2)
        x_train_poly = poly.fit_transform(train[:, 1:8])
        x_test_poly = poly.fit_transform(test[:, 1:8])

        # Train model
        model.fit(x_train_poly, train[:, 0])

        # Predict
        y_train_pred = model.predict(x_train_poly)
        y_test_pred = model.predict(x_test_poly)

        # Calculate MSE
        mse_train = mean_squared_error(train[:, 0], y_train_pred)
        mse_test = mean_squared_error(test[:, 0], y_test_pred)

        # Print results
        print(labels[i] + ":")
        print("MSE train: %.2f" % mse_train)
        print("MSE test: %.2f" % mse_test)
        print()

"""
Highest R^2 for each model:

Elastic net penalized linear regression with polynomial features:
MSE train: 11.40
MSE test: 15.62

Support Vector Regression (linear):
MSE train: 6.08
MSE test: 9.64

Support Vector Regression (polynomial):
MSE train: 17.34
MSE test: 22.98

Support Vector Regression (RBF):
MSE train: 11.04
MSE test: 19.83

Random Forest Regression:
MSE train: 1.00
MSE test: 10.28

K-Nearest Neighbors Regression (KNN):
MSE train: 5.57
MSE test: 12.84
"""
