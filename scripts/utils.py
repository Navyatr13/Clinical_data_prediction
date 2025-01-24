import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return torch.tensor(weights, dtype=torch.float32)


def handle_outliers(df, columns, method='clip'):
    """
    Handle outliers in specified columns.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == 'clip':
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        elif method == 'median':
            median_value = df[col].median()
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median_value
        elif method == 'log':
            df[col] = np.log1p(df[col])
    return df

def validate_preprocessed_data(df):
    """
    Perform validation checks on the preprocessed dataset.
    """
    print("Starting validation...")
    # Handle missing values in specific columns
    print("Filling missing values in specific columns...")
    fill_values = {'BaseExcess': 0, 'HCO3': 24}  # Domain-specific defaults
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # Check for remaining missing values
    print("Checking for remaining missing values...")
    missing_cols = df.columns[df.isnull().any()]
    if not missing_cols.empty:
        print(f"Columns with missing values: {list(missing_cols)}")
        print("Missing value counts per column:")
        print(df[missing_cols].isnull().sum())
        raise ValueError("There are still missing values in the dataset!")
    else:
        print("No missing values detected.")

    # Handle outliers
    print("Handling outliers...")
    numeric_cols = df.select_dtypes(include='number').columns
    df = handle_outliers(df, numeric_cols, method='clip')

    # Step 5: Validate label distributions
    print("Validating label distributions...")
    if 'InfectionLabel' in df.columns:
        print("InfectionLabel distribution:")
        print(df['InfectionLabel'].value_counts())
    else:
        print("InfectionLabel not found in the dataset.")

    if 'OrganDysfunctionLabel' in df.columns:
        print("OrganDysfunctionLabel distribution:")
        print(df['OrganDysfunctionLabel'].value_counts())
    else:
        print("OrganDysfunctionLabel not found in the dataset.")

    print("Validation completed successfully.")
    # Visualize distributions
    visualize_distributions(df, numeric_cols)

def visualize_distributions(df, numeric_cols):
    """
    Visualize distributions of numeric features and labels.
    """
    print("Visualizing distributions...")
    # Calculate correlation with SepsisLabel
    correlations = df[numeric_cols].corrwith(df['SepsisLabel']).sort_values(ascending=False)

    # Select top N correlated features
    top_correlated_cols = correlations.head(10).index.tolist()
    print("Top 10 columns correlated with SepsisLabel:", top_correlated_cols)

    for col in top_correlated_cols:
        plt.figure()
        df[col].hist(bins=50)
        plt.title(f"{col} Distribution")
        plt.show()

    # Label distributions
    if 'InfectionLabel' in df.columns:
        sns.countplot(x='InfectionLabel', data=df)
        plt.title("InfectionLabel Distribution")
        plt.show()

    if 'OrganDysfunctionLabel' in df.columns:
        sns.countplot(x='OrganDysfunctionLabel', data=df)
        plt.title("OrganDysfunctionLabel Distribution")
        plt.show()
