import pandas as pd

# Load the dataset
df = pd.read_csv("test_dataset01/AmesHousing.csv")

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
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")

# print("Path to dataset files:", path)