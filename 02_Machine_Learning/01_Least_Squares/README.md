# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```

Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: HEMASH K
RegisterNumber: 212225050015




import numpy as np
import matplotlib.pyplot as plt

# Input Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Compute Means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate numerator and denominator for slope
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

# Calculate slope (m)
m = numerator / denominator

# Calculate intercept (b)
b = y_mean - m * x_mean

# Regression line
y_pred = m * x + b

# Display Results
print("Mean of x =", x_mean)
print("Mean of y =", y_mean)
print("Slope (m) =", m)
print("Intercept (b) =", b)
print("Equation of Best Fit Line:")
print(f"y = {m:.2f}x + {b:.2f}")

# Plot Graph
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', linewidth=2, label='Best Fit Line')

plt.title("Univariate Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()
```

## Output:

<img width="570" height="422" alt="image" src="https://github.com/user-attachments/assets/09544146-3a65-4826-ae6f-c6a194bc6b07" />



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
