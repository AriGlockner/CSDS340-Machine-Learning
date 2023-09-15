from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Preprocess the data by standardizing it
scalar = StandardScaler()
X = scalar.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
