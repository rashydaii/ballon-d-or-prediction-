import pandas as pd

df = pd.read_csv("C:/Users/Acer/Downloads/archive (3)/ballondor_performance_vs_pr.csv")
print(df.head())
print(df.info())
print(df.describe())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



# Basic EDA
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# Visualizations
numeric_cols = df.select_dtypes(include="number").columns

plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Points"])
plt.title("Box Plot of Ballon d'Or Points")
plt.show()
if "Points" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Points"], kde=True)
    plt.title("Distribution of Ballon d'Or Points")
    plt.show()

# Feature Engineering
num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_cols = df.select_dtypes(exclude="number").columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# Save the processed data
df.to_csv("ballondor_processed.csv", index=False)

print("\nProcessed dataset shape:", df.shape)
print("Saved as ballondor_processed.csv")
print("Hona gay!")