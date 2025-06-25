[7:42 pm, 25/06/2025] Jay: import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Import the dataset
df = pd.read_csv("C:/Users/PAYAL MAHARANA/OneDrive/Documents/python/world_bank_data_2025.csv")

# Backup original data
df_original = df.copy()

# Preview the data
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Handle missing values
# Fill numeric columns with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Step 3: Encode categorical features using one-hot encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# STEP 4: Split into features and target (Assume 'target_column' is the target)
# You must change 'target_column' to your actual target column name
target_column = 'target_column'  # <--- CHANGE THIS
if target_column not in df.columns:
    raise ValueError("Please replace 'target_column' with your actual target column name.")

X = df.drop(target_column, axis=1)
y = df[target_column]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features (ONLY on training data to avoid data leakage)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# STEP 5: Visualize and Remove Outliers (IQR method on selected columns only)
# Replace below list with actual numeric features you care about for outlier detection
selected_outlier_cols = ['col1', 'col2']  # <--- CHANGE THIS
for col in selected_outlier_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (X_train[col] >= (Q1 - 1.5 * IQR)) & (X_train[col] <= (Q3 + 1.5 * IQR))
    
    # Apply same mask to X and y to keep consistency
    X_train = X_train[mask]
    y_train = y_train[mask]

# Optional: Boxplot after scaling
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train[selected_outlier_cols])
plt.xticks(rotation=45)
plt.title("Boxplot of Selected Numerical Features (After Scaling and Outlier Removal)")
plt.tight_layout()
plt.show()

# Final preview
print("\nFinal Cleaned Training Data:")
print(X_train.head())
print("\nFinal Cleaned Testing Data:")
print(X_test.head())