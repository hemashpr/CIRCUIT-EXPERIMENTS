# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Iris dataset.
2. Separate the input features and target species labels.
3. Split the dataset into training and testing data.
4. Train the SGD Classifier using the training data.
5. Predict the Iris species and calculate the accuracy of the model.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Hemash K
RegisterNumber: 212225050015
*/

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Input features
X = iris.data

# Target variable
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SGD Classifier
model = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:", accuracy * 100, "%")
```

## Output:
<img width="648" height="117" alt="image" src="https://github.com/user-attachments/assets/ec80ae5a-14bc-4d1a-8f35-9b8e2b268519" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
