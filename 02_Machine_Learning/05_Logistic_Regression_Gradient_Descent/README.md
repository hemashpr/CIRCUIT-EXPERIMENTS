# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Placement dataset.
2. Preprocess the data and convert categorical values into numerical form.
3. Initialize weights and apply the Sigmoid function to calculate predictions.
4. Compute the cost and update weights using Gradient Descent.
5. Repeat for the required iterations and calculate the model accuracy.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Hemash K
RegisterNumber:  212225050015
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Placement_Data.csv")

print("Original Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Remove unnecessary serial number column if present
if "sl_no" in df.columns:
    df = df.drop("sl_no", axis=1)

# Target column
target_col = "status"

# IMPORTANT:
# Salary is usually missing for students who are Not Placed.
# Remove Salary instead of dropping all rows containing missing values.
if "salary" in df.columns:
    df = df.drop("salary", axis=1)

# Remove rows only if the TARGET itself is missing
df = df.dropna(subset=[target_col])

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

print("\nTarget Classes:")
print(y.unique())

# Convert target to binary
y = y.map({
    "Placed": 1,
    "Not Placed": 0
})

# Convert categorical features to numerical values
X = pd.get_dummies(X, drop_first=True)

# Convert to NumPy arrays
X = X.astype(float).values
y = y.values.astype(float)

# Feature scaling
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

std[std == 0] = 1

X = (X - mean) / std

# Add bias column
m = X.shape[0]
X = np.c_[np.ones((m, 1)), X]

# Sigmoid Function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# Cost Function
def compute_cost(X, y, theta):

    m = len(y)

    h = sigmoid(np.dot(X, theta))

    epsilon = 1e-10

    cost = -(1 / m) * np.sum(
        y * np.log(h + epsilon) +
        (1 - y) * np.log(1 - h + epsilon)
    )

    return cost


# Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):

    m = len(y)
    cost_history = []

    for i in range(iterations):

        # Calculate predictions
        h = sigmoid(np.dot(X, theta))

        # Calculate gradient
        gradient = (1 / m) * np.dot(X.T, (h - y))

        # Update parameters
        theta = theta - learning_rate * gradient

        # Calculate cost
        cost = compute_cost(X, y, theta)

        cost_history.append(cost)

        if i % 100 == 0:
            print("Iteration:", i, "Cost:", round(cost, 6))

    return theta, cost_history


# Initialize parameters
theta = np.zeros(X.shape[1])

# Set learning rate and iterations
learning_rate = 0.1
iterations = 1000

# Train model
theta, cost_history = gradient_descent(
    X, y, theta, learning_rate, iterations
)

# Calculate probabilities
probabilities = sigmoid(np.dot(X, theta))

# Convert probabilities to classes
predictions = (probabilities >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions == y) * 100

print("\n-----------------------------")
print("FINAL RESULTS")
print("-----------------------------")
print("Final Cost:", cost_history[-1])
print("Accuracy: {:.2f}%".format(accuracy))

# Display predictions
results = pd.DataFrame({
    "Actual": y.astype(int),
    "Predicted": predictions,
    "Probability": np.round(probabilities, 4)
})

print("\nFirst 10 Predictions:")
print(results.head(10))

# Plot Cost vs Iterations
plt.figure(figsize=(8, 5))
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Logistic Regression Using Gradient Descent")
plt.grid()
plt.show()
```

## Output:
<img width="1038" height="406" alt="image" src="https://github.com/user-attachments/assets/f06c6795-940a-42d6-bec1-a966ca46f43e" />
<img width="1017" height="716" alt="image" src="https://github.com/user-attachments/assets/a71ae51f-d668-4618-90d7-30375d170461" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

