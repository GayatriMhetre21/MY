import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids

# Load the credit card dataset
url = 'https://www.kaggle.com/datasets/arjunbhasin2013/ccdata'
# Note: You need to download the dataset from Kaggle and specify the correct path here
data = pd.read_csv('ccdata.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data: Drop non-numeric columns and handle missing values if necessary
data = data.drop(columns=['CUST_ID'])  # Drop customer ID or any other non-numeric columns

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Determine the optimal number of clusters using Silhouette Method
silhouette_scores = []
k_range = range(2, 11)  # Testing cluster sizes from 2 to 10

for k in k_range:
    # Initialize K-Medoids with random medoids
    initial_medoids = np.random.choice(len(X_scaled), k, replace=False)
    kmedoids_instance = kmedoids(X_scaled, initial_medoids)
    kmedoids_instance.process()
    
    # Get cluster labels
    cluster_labels = kmedoids_instance.get_clusters()
    
    # Calculate silhouette score (need to flatten cluster labels)
    labels = np.zeros(len(X_scaled))
    for cluster_id, indices in enumerate(cluster_labels):
        labels[indices] = cluster_id
    
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plotting silhouette scores to determine optimal k
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid()
plt.show()

# Based on the silhouette plot, choose an appropriate number of clusters (e.g., 3)
optimal_k = 3  # Adjust this based on your silhouette plot observation

# Fit K-Medoids with the optimal number of clusters
initial_medoids = np.random.choice(len(X_scaled), optimal_k, replace=False)
kmedoids_final = kmedoids(X_scaled, initial_medoids)
kmedoids_final.process()

# Get final cluster labels
final_clusters = kmedoids_final.get_clusters()

# Display resulting clusters
data['Cluster'] = -1  # Initialize with -1 for unassigned points
for cluster_id, indices in enumerate(final_clusters):
    data.loc[indices, 'Cluster'] = cluster_id

print(data[['Cluster']].head(10))


