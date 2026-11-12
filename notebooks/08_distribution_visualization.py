import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("graphs", exist_ok=True)
# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_aqi.csv")

df["date"] = pd.to_datetime(df["date"])

# 2. Split training and testing data
unique_dates = sorted(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

# 3. Select four cities
cities = ["Jaipur", "Jodhpur", "Delhi", "Agra"]

train_df = train_df[train_df["city"].isin(cities)].copy()
test_df = test_df[test_df["city"].isin(cities)].copy()

# 4. Add dataset type
train_df["Dataset"] = "Training"
test_df["Dataset"] = "Testing"

# Combine both datasets
combined_df = pd.concat([train_df, test_df])

sns.set_theme(style="whitegrid")

# 5. PM10 distribution comparison
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=combined_df,
    x="city",
    y="pm10_mean",
    hue="Dataset"
)

plt.title("Training vs Testing - PM10 Distribution")
plt.xlabel("City")
plt.ylabel("PM10 Mean")
plt.tight_layout()
plt.savefig("graphs/09_PM10_distribution_comparison.png",dpi = 300)
plt.show()

# 6. AQI distribution comparison
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=combined_df,
    x="city",
    y="india_aqi",
    hue="Dataset"
)

plt.title("Training vs Testing - AQI Distribution")
plt.xlabel("City")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("graphs/10_AQI_distribution_comparison.png",dpi = 300)
plt.show()

# 7. PM10 vs AQI relationship
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=test_df,
    x="pm10_mean",
    y="india_aqi",
    hue="city",
    alpha=0.6
)

plt.title("PM10 vs AQI - Testing Data")
plt.xlabel("PM10 Mean")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("graphs/11_PM10_vs_AQI_relationship.png",dpi = 300)
plt.show()

print("\n===== DISTRIBUTION VISUALIZATION COMPLETED =====")
