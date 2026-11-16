# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Employee dataset using Pandas.
2. Preprocess the dataset by handling missing values and converting categorical data into numerical form.
3. Separate the input features and target variable for employee prediction.
4. Split the dataset into training and testing sets and train the Decision Tree Classifier.
5. Predict the test results and evaluate the model using accuracy, classification report, and confusion matrix.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Hemash K
RegisterNumber:  212225050015
*/

from google.colab import files

uploaded = files.upload()

# Program to implement the Decision Tree Classifier Model
# for Predicting Employee Churn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("Employee.csv")

# Display the first five records
print(data.head())

# Display dataset information
print(data.info())

# Remove missing values if any
data = data.dropna()

# Convert categorical columns into numerical values
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Display the processed dataset
print(data.head())


# Select the target column
X = data.drop("salary", axis=1)
y = data["salary"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy of Decision Tree Classifier:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Output:
<img width="867" height="626" alt="image" src="https://github.com/user-attachments/assets/1a993db1-b817-49a0-97e6-78fd25d5684e" />
<img width="822" height="560" alt="image" src="https://github.com/user-attachments/assets/66a88ef3-71a6-40cd-b13c-b87c9873e44d" />
<img width="771" height="85" alt="image" src="https://github.com/user-attachments/assets/59e0a1d3-9a19-42dc-b532-b0d989834869" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
