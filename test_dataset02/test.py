import pandas as pd

# Load the dataset
df = pd.read_csv("test_dataset02/train.csv")

# ==============================
# Basic Information
# ==============================
print("\nDataset Shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include='all'))

# ==============================
# Directly Download Dataset from Kaggle
# ==============================

# Please ensure you have the Kaggle API installed and configured with your API credentials.
# I cant upload the train.csv flie due to License issues. You can download the dataset directly from Kaggle using the following code snippet.

# import kagglehub

# # Download latest version
# path = kagglehub.competition_download('house-prices-advanced-regression-techniques')

# print("Path to competition files:", path)