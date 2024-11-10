import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
url = 'https://www.kaggle.com/datasets/rakeshrau/social-network-ads'
# Note: You need to download the dataset from Kaggle and specify the correct path here
data = pd.read_csv('Social_Network_Ads.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Drop unnecessary columns (if any)
data = data.drop(columns=['User ID'])

# Convert categorical variable 'Gender' to numerical
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Define features (X) and target (y)
X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors as needed
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Compute evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Display results
print("Confusion Matrix:\n", conf_matrix)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Error Rate: {error_rate * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')

