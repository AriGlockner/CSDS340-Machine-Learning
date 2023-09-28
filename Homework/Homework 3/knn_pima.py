"""
Train a k-nearest neighbor classifier to the Pima Indians data. Split the data 50/50 into training and tests sets using
train_test_split() with random_state=1 to provide a fixed seed. Try out different approaches for feature scaling and
selection to see their effects on the test set classification accuracy. Report the highest test set accuracy you are
able to obtain.

@author: ari
"""
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Load the Pima Indians dataset
pima = read_csv('pima-indians-diabetes.csv')

# Split the data into training and test sets
X = pima.drop('diabetes', axis=1)
y = pima['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a k-nearest neighbor classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the test set labels
y_pred = knn.predict(X_test)

# Calculate the test set accuracy
print('Test set accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
