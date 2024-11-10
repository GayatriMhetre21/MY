import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
url = 'https://www.kaggle.com/datasets/uciml/iris'
# Note: You need to download the dataset from Kaggle and specify the correct path here
data = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
print(data.head())

# Drop the 'Id' column and set features for clustering
X = data.drop(columns=['Id', 'Species'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)  # Testing cluster sizes from 1 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid()
plt.show()

# Based on the elbow plot, choose an appropriate number of clusters (e.g., 3)
optimal_k = 3  # Adjust this based on your elbow plot observation

# Fit K-Means with the optimal number of clusters
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Display the resulting clusters
print(data[['Species', 'Cluster']].head(10))

