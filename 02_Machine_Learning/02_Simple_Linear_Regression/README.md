# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries (pandas, numpy, matplotlib, and scikit-learn modules).
2. Load the student scores dataset.
3. Prepare the input feature (Hours) and target output (Scores).
4. Split the dataset into training and testing sets.
5. Create and train the Simple Linear Regression model.
6. Predict the marks for the test dataset.
7. Plot the regression line along with the data points.
8. Evaluate the model using performance metrics (MAE, MSE, RMSE).
9. Predict the marks for new input samples.
 
  
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
# Developed by: HEMASH K
# Register Number: 212225050015 
*/

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load Dataset 


df = pd.read_csv("student_scores.csv")

print("First 5 Rows:\n")
print(df.head())

print("\nLast 5 Rows:\n")
print(df.tail())


# Prepare Input (X) and Output (Y)

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

print("\nX (Features):")
print(X.flatten())

print("\nY (Targets):")
print(Y)


# Split Dataset


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=1/3,
    random_state=0
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# Train Model

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

print("\nModel Trained Successfully!")

print("Slope (Coefficient):", regressor.coef_[0])
print("Intercept:", regressor.intercept_)


# Prediction

Y_pred = regressor.predict(X_test)

print("\nActual Values:")
print(Y_test)

print("\nPredicted Values:")
print(np.round(Y_pred,2))


# Training Graph

plt.figure(figsize=(7,5))

plt.scatter(X_train, Y_train, color='orange', label='Training Data')
plt.plot(X_train,
         regressor.predict(X_train),
         color='red',
         label='Regression Line')

plt.title("Training Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)

plt.show()


# Testing Graph


order = np.argsort(X_test.flatten())

X_test_sorted = X_test.flatten()[order]
Y_pred_sorted = Y_pred[order]

plt.figure(figsize=(7,5))

plt.scatter(X_test, Y_test, color='blue', label='Testing Data')
plt.plot(X_test_sorted,
         Y_pred_sorted,
         color='green',
         label='Regression Line')

plt.title("Testing Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)

plt.show()


# Evaluation Metrics

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\nMean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("Root Mean Squared Error :", rmse)


# Predict New Student Marks

new_hours = np.array([[2.5],
                      [5.0],
                      [9.25]])

predicted_marks = regressor.predict(new_hours)

print("\nPredicted Marks")

for hrs, marks in zip(new_hours.flatten(), predicted_marks):
    print(f"{hrs} Hours --> {marks:.2f} Marks")
```

## Output:

<img width="1842" height="755" alt="image" src="https://github.com/user-attachments/assets/f604c123-ba69-44e3-99a2-a6b6839eb0a4" />


<img width="1811" height="757" alt="image" src="https://github.com/user-attachments/assets/49883d33-e553-4434-b27d-90a1ddc5cd41" />







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
