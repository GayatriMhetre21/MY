import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "https://www.kaggle.com/datasets/yasserh/uber-fares-dataset"
dataset_path = "uber.csv"  # Make sure the file is downloaded and the path is correct
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(data.head())

# 1. Pre-process the dataset
# Convert 'pickup_datetime' to datetime format
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

# Extract date features
data['year'] = data['pickup_datetime'].dt.year
data['month'] = data['pickup_datetime'].dt.month
data['day'] = data['pickup_datetime'].dt.day
data['hour'] = data['pickup_datetime'].dt.hour
data['minute'] = data['pickup_datetime'].dt.minute

# Drop unnecessary columns
data.drop(['pickup_datetime', 'pickup_date'], axis=1, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Convert categorical columns to numerical
data = pd.get_dummies(data, columns=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])

# 2. Identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['fare_amount'])
plt.title('Boxplot of Fare Amount')
plt.show()

# Remove outliers
q1 = data['fare_amount'].quantile(0.25)
q3 = data['fare_amount'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data = data[(data['fare_amount'] >= lower_bound) & (data['fare_amount'] <= upper_bound)]

# 3. Check the correlation
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 4. Implement linear regression, ridge regression, and Lasso regression models
# Define features and target variable
X = data.drop('fare_amount', axis=1)
y = data['fare_amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)

# Train the models
linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
lasso_model.fit(X_train_scaled, y_train)

# Predict using the models
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_lasso = lasso_model.predict(X_test_scaled)

# 5. Evaluate the models and compare their respective scores
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

r2_linear, rmse_linear = evaluate_model(y_test, y_pred_linear)
r2_ridge, rmse_ridge = evaluate_model(y_test, y_pred_ridge)
r2_lasso, rmse_lasso = evaluate_model(y_test, y_pred_lasso)

print(f'Linear Regression: R² = {r2_linear:.2f}, RMSE = {rmse_linear:.2f}')
print(f'Ridge Regression: R² = {r2_ridge:.2f}, RMSE = {rmse_ridge:.2f}')
print(f'Lasso Regression: R² = {r2_lasso:.2f}, RMSE = {rmse_lasso:.2f}')

