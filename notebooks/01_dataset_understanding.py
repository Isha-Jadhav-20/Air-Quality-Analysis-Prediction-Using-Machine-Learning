
import pandas as pd
import numpy as np

# 1. Load Dataset
df = pd.read_csv("dataset/air_quality_daily.csv")

# 2. First 5 Records
print("\n===== FIRST 5 RECORDS =====")
print(df.head())

# 3. Dataset Shape
print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 4. Column Names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# 5. Dataset Information
print("\n===== DATASET INFORMATION =====")
df.info()

# 6. Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 7. Statistical Summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# 8. Duplicate Records
print("\n===== DUPLICATE RECORDS =====")
print(df.duplicated().sum())

# 9. Date Range
df["date"] = pd.to_datetime(df["date"])

print("\n===== DATE RANGE =====")
print("Start Date:", df["date"].min())
print("End Date:", df["date"].max())

# 10. Unique Cities
print("\n===== TOTAL CITIES =====")
print(df["city"].nunique())

print("\n===== CITY NAMES =====")
print(df["city"].unique())

print("\n===== DATASET LOADED SUCCESSFULLY =====")
