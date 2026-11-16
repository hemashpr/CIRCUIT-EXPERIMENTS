# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Create the dataset containing house area, bedrooms, price, and occupants.
2. Separate the input features and target values, then split the data into training and testing sets.
3. Standardize the input features using StandardScaler.
4. Train SGDRegressor models using the training data to predict house price and number of occupants.
5. Test the models and predict the price and occupants for a new house.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Hemash K
RegisterNumber:  212225050015
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# Create dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 5],
    'Occupants': [3, 4, 4, 5, 5, 6, 6, 7],
    'Price': [30, 35, 45, 55, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

# Input features
X = df[['Area', 'Bedrooms']]

# Target values
y_price = df['Price']
y_occupants = df['Occupants']

# Split data
X_train, X_test, y_price_train, y_price_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

# Use the same indices for occupants
_, _, y_occ_train, y_occ_test = train_test_split(
    X, y_occupants, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


price_model = SGDRegressor(
    max_iter=1000,
    learning_rate='constant',
    eta0=0.01,
    random_state=42
)

price_model.fit(X_train_scaled, y_price_train)

price_prediction = price_model.predict(X_test_scaled)



occupant_model = SGDRegressor(
    max_iter=1000,
    learning_rate='constant',
    eta0=0.01,
    random_state=42
)

occupant_model.fit(X_train_scaled, y_occ_train)

occupant_prediction = occupant_model.predict(X_test_scaled)



print("Actual House Prices:")
print(y_price_test.values)

print("\nPredicted House Prices:")
print(price_prediction)

print("\nActual Number of Occupants:")
print(y_occ_test.values)

print("\nPredicted Number of Occupants:")
print(occupant_prediction)


new_house = [[2000, 3]]

new_house_scaled = scaler.transform(new_house)

new_price = price_model.predict(new_house_scaled)
new_occupants = occupant_model.predict(new_house_scaled)

print("\nFor a new house:")
print("Area       : 2000 sq.ft")
print("Bedrooms   : 3")
print("Predicted Price:", new_price[0], "lakhs")
print("Predicted Number of Occupants:", round(new_occupants[0]))
```

## Output:

<img width="342" height="240" alt="image" src="https://github.com/user-attachments/assets/e849fd6a-767f-45ad-87d4-5e2ec1f4765c" />
<img width="345" height="167" alt="image" src="https://github.com/user-attachments/assets/3aa94a60-3652-42d6-8dde-9e2e8a3e4332" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
