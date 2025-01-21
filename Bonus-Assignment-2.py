
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Step 3: Check for missing values and handle them
print("
Checking for missing values:")
print(iris_df.isnull().sum())

# Simulate missing values (if needed) and handle them
iris_df.iloc[0, 0] = np.nan  # Introduce a missing value for demonstration
print("
After introducing a missing value:")
print(iris_df.isnull().sum())

# Fill missing values with the column mean
iris_df.fillna(iris_df.mean(), inplace=True)
print("
After handling missing values:")
print(iris_df.isnull().sum())

# Step 4: Perform basic EDA
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()  # Ensure layout is adjusted
plt.show()

# Visualize the distribution of numerical columns
for column in iris.feature_names:
    plt.figure(figsize=(6, 4))
    sns.histplot(iris_df[column], kde=True, bins=20, color='blue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()  # Adjust spacing for each individual plot
    plt.show()

# Step 5: Export cleaned data to a CSV file
cleaned_file_path = 'cleaned_iris_data.csv'
iris_df.to_csv(cleaned_file_path, index=False)
print(f"
Cleaned data exported to: {cleaned_file_path}")
