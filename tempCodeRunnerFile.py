import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



#load the dataset
df = pd.read_csv("C:/Users/Acer/Downloads/archive (3)/ballondor_performance_vs_pr.csv")

# Exploratory Data Analysis (EDA)
# Display first five rows
print("First 5 Rows")
print(df.head())

# Dataset shape
print("\nDataset Shape")
print(df.shape)

# Dataset information
print("\nDataset Information")
df.info()

# Descriptive statistics    
print("\nDescriptive Statistics")
print(df.describe())

# Missing Value Analysis

print("Missing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

# 4. Data Visualization

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.show()

# Box Plot
plt.figure(figsize=(8,4))
sns.boxplot(x=df["Points"])
plt.title("Box Plot of Ballon d'Or Points", fontsize=14)
plt.show()

# Distribution Plot
plt.figure(figsize=(8,4))
sns.histplot(df["Points"], kde=True)
plt.title("Top 10 Players by Ballon d'Or Points", fontsize=14)
plt.xlabel("Ballon d'Or Points")
plt.ylabel("Player")
plt.show()

# Feature Engineering

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Handle missing values using median
numeric_cols = df.select_dtypes(include="number").columns

imputer = SimpleImputer(strategy="median")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("Missing Values After Imputation")
print(df.isnull().sum())

# One-Hot Encode categorical columns
categorical_cols = df.select_dtypes(exclude="number").columns

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scale numerical features (excluding target variable)
scaler = StandardScaler()

feature_cols = df.columns.drop("Points")

df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("\nProcessed Dataset")
print(df.head())



# 6. Save the Processed Dataset


df.to_csv("ballondor_processed.csv", index=False)

print("Processed dataset saved successfully!")