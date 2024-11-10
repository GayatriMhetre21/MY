import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
url = 'https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set'
# Note: You need to download the dataset from Kaggle and specify the correct path here
data = pd.read_csv('car_evaluation.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# The dataset contains categorical features; we need to convert them to numerical values
data['buying'] = data['buying'].map({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})
data['maint'] = data['maint'].map({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})
data['doors'] = data['doors'].replace({'2': 2, '3': 3, '4': 4, '5more': 5})
data['persons'] = data['persons'].replace({'2': 2, '4': 4, 'more': 5})
data['lug_boot'] = data['lug_boot'].map({'small': 1, 'med': 2, 'big': 3})
data['safety'] = data['safety'].map({'low': 1, 'med': 2, 'high': 3})

# Define features (X) and target (y)
X = data.drop(columns=['class'])  # Features
y = data['class']                 # Target variable (safety class)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')


