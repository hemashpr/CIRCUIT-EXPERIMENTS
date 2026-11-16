# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and load the Placement dataset.
2. Remove unnecessary columns and handle categorical data using encoding.
3. Separate the input features (X) and placement status (y).
4. Split the dataset into training and testing sets and train the Logistic Regression model.
5. Predict the placement status and evaluate the model using accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Hemash K
RegisterNumber: 212225050015 
*/

# Implementation of Logistic Regression Model
# To Predict the Placement Status of Student

# Step 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 2: Load the dataset
data = pd.read_csv("Placement_Data.csv")


# Step 3: Display the dataset
print("Dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())


# Step 4: Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Step 5: Remove unnecessary columns
if 'sl_no' in data.columns:
    data = data.drop('sl_no', axis=1)

if 'salary' in data.columns:
    data = data.drop('salary', axis=1)


# Step 6: Convert categorical columns into numerical values
label_encoder = LabelEncoder()

for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])


# Step 7: Separate independent and dependent variables
X = data.drop('status', axis=1)
y = data['status']


# Step 8: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 9: Create Logistic Regression model
model = LogisticRegression(max_iter=1000)


# Step 10: Train the model
model.fit(X_train, y_train)


# Step 11: Predict the placement status
y_pred = model.predict(X_test)


# Step 12: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy of Logistic Regression Model:")
print(accuracy * 100, "%")


# Step 13: Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Step 14: Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Output:
<img width="498" height="592" alt="image" src="https://github.com/user-attachments/assets/d95642a6-ba92-4a55-b5cc-4ba16787dfa4" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
