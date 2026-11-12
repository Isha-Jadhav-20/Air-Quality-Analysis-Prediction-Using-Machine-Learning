import pandas as pd
import numpy as np

# 1) Load Dataset
df = pd.read_csv("dataset/air_quality_daily.csv")

print("Original Dataset Shape:", df.shape)

# 2) Select Required Columns
selected_columns = [
    "date",
    "city",
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max",
    "india_aqi"
]

df = df[selected_columns].copy()

# 3) Convert Date Column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 4) Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

df = df.drop_duplicates(
    subset=["date", "city"]
)

df = df.reset_index(drop=True)

# 7) Check Negative Pollutant Values
pollutant_columns = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max"
]

# Remove invalid negative pollutant values
df = df[(df[pollutant_columns] >= 0).all(axis=1)]

# 8) Check Valid AQI Range
df = df[
    (df["india_aqi"] >= 0) &
    (df["india_aqi"] <= 500)
]

# 9) Sort Dataset
df = df.sort_values(["date", "city"])

# 10) Reset Index
df.reset_index(drop=True, inplace=True)

# 11) Display Clean Dataset
print("\n----- CLEANED DATASET -----")
print(df.head())

print("\nCleaned Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records After Cleaning:")
print(df.duplicated(subset=["date", "city"]).sum())

# 12) Save Cleaned Dataset
df.to_csv("dataset/cleaned_aqi.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
