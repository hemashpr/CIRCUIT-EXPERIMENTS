# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the customer dataset using Pandas.
2, Select relevant customer features and preprocess the data for clustering.
3, Apply the K-Means algorithm by specifying the required number of clusters.
4, Train the model and assign each customer to the identified cluster.
5, Visualize and analyze the clusters for effective customer segmentation.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: hemash k
RegisterNumber:  212225050015
*/

# Implementation of K-Means Clustering for Customer Segmentation

import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.cluster import KMeans

# Upload the dataset
uploaded = files.upload()

# Load the uploaded dataset
filename = list(uploaded.keys())[0]
data = pd.read_csv(filename)

# Display the first five records
print("Dataset:")
print(data.head())

# Display column names
print("\nColumns:")
print(data.columns.tolist())

# Select numerical columns for clustering
X = data.select_dtypes(include=['int64', 'float64'])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model and predict clusters
data['Cluster'] = kmeans.fit_predict(X)

# Display customer data with cluster labels
print("\nCustomer Clusters:")
print(data.head())

# Visualize the clusters using the first two numerical features
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=data['Cluster']
)

plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("Customer Segmentation using K-Means Clustering")

plt.show()
```

## Output:

<img width="942" height="773" alt="image" src="https://github.com/user-attachments/assets/fd9f954c-d5dc-42ab-9ed5-52228273eecd" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
