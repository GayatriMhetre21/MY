import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the Iris dataset
url = 'https://www.kaggle.com/datasets/uciml/iris'
# Note: You need to download the dataset from Kaggle and specify the correct path here
data = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
X = data.drop(columns=['Id', 'Species'])  # Features
y = data['Species']                        # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100)

# Train classifiers
ada_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Make predictions
ada_pred = ada_model.predict(X_test)
gbm_pred = gbm_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print(f'Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%\n')

print("AdaBoost Performance:")
evaluate_model(y_test, ada_pred)

print("Gradient Boosting Performance:")
evaluate_model(y_test, gbm_pred)

print("XGBoost Performance:")
evaluate_model(y_test, xgb_pred)


