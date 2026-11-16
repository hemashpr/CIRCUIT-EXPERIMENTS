# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the environmental sensor dataset using Pandas.
2. Preprocess the data by handling missing values and selecting relevant atmospheric and environmental features.
3. Separate the input features and target variables such as temperature, PM2.5 level, and energy.
4. Split the dataset into training and testing sets and train a Random Forest Regression model.
5. Evaluate the model using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: hemash k
RegisterNumber:  212225050015
*/

# Implementation of Random Forest Algorithm for Weather Prediction

# Import required libraries
import pandas as pd
import numpy as np
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Upload the dataset
uploaded = files.upload()

# Get the uploaded filename
filename = list(uploaded.keys())[0]

# Load the dataset
data = pd.read_csv(filename)

# Display the first five records
print("First five records of the dataset:\n")
print(data.head())

# Display column names
print("\nColumns in the dataset:\n")
print(data.columns.tolist())

# Display dataset information
print("\nDataset Information:\n")
print(data.info())

# Remove missing values
data = data.dropna()

# Select numerical columns
numeric_data = data.select_dtypes(include=np.number)

print("\nNumerical Columns:\n")
print(numeric_data.columns.tolist())

# Select target variable
# The last numerical column is used as the prediction target
target_column = numeric_data.columns[-1]

print("\nTarget Variable:", target_column)

# Separate input features and target variable
X = numeric_data.drop(target_column, axis=1)
y = numeric_data[target_column]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

# Display results
print("\n----- MODEL PERFORMANCE -----")

print("R2 Score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Display actual and predicted values
results = pd.DataFrame({
    "Actual Values": y_test.values,
    "Predicted Values": y_pred
})

print("\n----- ACTUAL VS PREDICTED VALUES -----")
print(results.head(10))
```

## Output:
<img width="1101" height="437" alt="image" src="https://github.com/user-attachments/assets/53520268-e6c8-494f-8cf1-456d8cfb5b26" />

<img width="1043" height="508" alt="image" src="https://github.com/user-attachments/assets/26e3c0aa-1d33-489e-b897-93354c52cf48" />



## Result:

Thus, the Random Forest Regression model was successfully implemented and evaluated for predicting weather-related environmental parameters.

