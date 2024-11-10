'''
Viva Questions
1) Explain LDA?


'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# load the datset
iris_df = pd.read_csv("Iris.csv")

# Separate the features and the target
X = iris_df.drop(['Id', 'Species'], axis=1)
y = iris_df['Species']

# Split the dataset into training and testing sets
# Here, 30% of the data is allocated for testing, and a random state is set for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply LDA
lda = LDA(n_components=2)  # Reduce to 2 components for visualization
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# Train a classifier (LDA itself can be used as a classifier)
lda_classifier = LDA()
lda_classifier.fit(X_train_lda, y_train)
y_pred = lda_classifier.predict(X_test_lda)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Visualize the LDA components
plt.figure(figsize=(10, 7))
for species in iris_df['Species'].unique():
    plt.scatter(X_train_lda[y_train == species, 0], 
                X_train_lda[y_train == species, 1], label=species)

plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA of Iris Dataset')
plt.legend()
plt.show()

print(accuracy, conf_matrix, class_report)




