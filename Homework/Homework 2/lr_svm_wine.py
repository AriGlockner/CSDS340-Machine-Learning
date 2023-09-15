from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Preprocess the data by standardizing it
scalar = StandardScaler()
X = scalar.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
c = 0.1

max_accuracy = 0.0
best_kernel = ''
best_c = 0.0

for kernel in kernels:
    for i in range(10):
        # Fit SVM model
        svm = SVC(kernel=kernel, C=c, random_state=1)
        svm.fit(X_train, y_train)

        # Check the accuracy of the model
        accuracy = accuracy_score(y_test, svm.predict(X_test))
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_kernel = kernel
            best_c = c

        c += 0.1

print(f"The best kernel is {best_kernel} with a C value of {best_c:.1f} and an accuracy of {100*max_accuracy:.2f}%")
