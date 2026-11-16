# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and load the startup dataset into the program.
2. Select R&D Spend as the input feature and Profit as the output variable, then normalize the data for better learning.
3. Initialize the slope and intercept, and use the Gradient Descent algorithm to update them iteratively until the model learns the best-fit line.
4. Use the trained model to predict the profit and plot the regression line along with the original data to visualize the results.
 
  
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Hemash k
RegisterNumber:  212225050015
*/
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("exp_3_50_Startups.csv")

# Independent and dependent variables
X = data["R&D Spend"].values
y = data["Profit"].values

# Feature Scaling (Normalization)
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()

# Initialize parameters
m = 0
c = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

# Gradient Descent
for i in range(epochs):

    y_pred = m * X + c

    dm = (-2/n) * np.sum(X * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

print("Slope (m):", m)
print("Intercept (c):", c)

# Predictions
y_pred = m * X + c

# Plot
plt.figure(figsize=(8,5))

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")

plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Simple Linear Regression using Gradient Descent")
plt.legend()

plt.show()
```

## Output:

<img width="707" height="487" alt="image" src="https://github.com/user-attachments/assets/9542b893-aa5c-4a89-85bb-51c973a6e280" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
