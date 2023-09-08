from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score


def print_accuracy(test, prediction, model_name):
    """
    Prints the accuracy of the model

    :param test:
    :param prediction:
    :param model_name: is this a Gaussian or Bernoulli Naïve Bayes model?
    """
    # Calculate the test set accuracy
    accuracy = accuracy_score(test, prediction)
    print(f"The " + model_name + f" accuracy is: {accuracy:.2f}")
    pass


# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Preprocess the data by standardizing it
scalar = StandardScaler()
X = scalar.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Fit a Gaussian Naïve Bayes classifier to the training data
gauss = GaussianNB()
gauss.fit(X_train, y_train)

# Fit a Bernoulli Naïve Bayes classifier to the training data
bern = BernoulliNB(force_alpha=True)
bern.fit(X_train, y_train)

# Make predictions on the test data
gaussian_predict = gauss.predict(X_test)
bernoulli_predict = bern.predict(X_test)

# Print the accuracy of the models
print_accuracy(y_test, gaussian_predict, "Gaussian Naïve Bayes")
print_accuracy(y_test, bernoulli_predict, "Gaussian Naïve Bayes")
