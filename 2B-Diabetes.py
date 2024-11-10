import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the Pima Indians Diabetes dataset
pima_df = pd.read_csv('path_to_pima_diabetes.csv')

# Univariate Analysis
def univariate_analysis(df):
    print("Univariate Analysis:")
    
    # Calculate statistics
    mean = df.mean()
    median = df.median()
    mode = df.mode().iloc[0]
    variance = df.var()
    std_dev = df.std()
    skewness = df.skew()
    kurtosis = df.kurtosis()

    # Display results
    print("\nMean:\n", mean)
    print("\nMedian:\n", median)
    print("\nMode:\n", mode)
    print("\nVariance:\n", variance)
    print("\nStandard Deviation:\n", std_dev)
    print("\nSkewness:\n", skewness)
    print("\nKurtosis:\n", kurtosis)

# Bivariate Analysis - Linear Regression
def linear_regression(df):
    print("\nBivariate Analysis - Linear Regression:")
    
    # Define independent and dependent variables
    X = df[['Glucose', 'BMI', 'Age']]  # Example predictors
    y = df['Outcome']

    # Add constant to predictor variables
    X = sm.add_constant(X)

    # Fit linear regression model
    model = sm.OLS(y, X).fit()
    
    # Print model summary
    print(model.summary())

# Bivariate Analysis - Logistic Regression
def logistic_regression(df):
    print("\nBivariate Analysis - Logistic Regression:")
    
    # Define independent and dependent variables
    X = df[['Glucose', 'BMI', 'Age']]
    y = df['Outcome']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit logistic regression model
    log_model = LogisticRegression(max_iter=200)
    log_model.fit(X_train, y_train)

    # Predict probabilities on the test set
    predictions = log_model.predict_proba(X_test)[:, 1]
    
    # Print coefficients and accuracy
    print("Logistic Regression Coefficients:\n", log_model.coef_)
    
# Multiple Regression Analysis
def multiple_regression(df):
    print("\nMultiple Regression Analysis:")
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Add constant to predictor variables
    X = sm.add_constant(X)

    # Fit multiple regression model
    multi_model = sm.OLS(y, X).fit()
    
    # Print model summary
    print(multi_model.summary())

# Comparison with another dataset (hypothetical)
def compare_datasets(pima_df, other_df):
    print("\nComparison of Datasets:")
    
    pima_mean = pima_df.mean()
    other_mean = other_df.mean()
    
    mean_comparison = pd.DataFrame({
        'Pima': pima_mean,
        'Other': other_mean
    })

    print("Mean Comparison:\n", mean_comparison)

# Main Function to Execute All Analyses
def main():
    univariate_analysis(pima_df)
    
    linear_regression(pima_df)
    
    logistic_regression(pima_df)
    
    multiple_regression(pima_df)

# Load another diabetes dataset for comparison (replace with actual path)
other_df = pd.read_csv('path_to_other_diabetes_dataset.csv')

if __name__ == "__main__":
     main()
     compare_datasets(pima_df, other_df)

