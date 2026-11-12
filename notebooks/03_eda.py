import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("graphs", exist_ok=True)
# Load Cleaned Dataset
df = pd.read_csv("dataset/cleaned_aqi.csv")

df["date"] = pd.to_datetime(df["date"])

sns.set_theme(style="whitegrid")

# 1. AQI Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["india_aqi"], bins=30, kde=True)
plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graphs/01_aqi_distribution.png",dpi = 300)
plt.show()

# 2. Top 10 Cities by Average AQI
city_aqi = (
    df.groupby("city")["india_aqi"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
sns.barplot(x=city_aqi.values, y=city_aqi.index)
plt.title("Top 10 Cities by Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("graphs/02_top_10_cities.png",dpi = 300)
plt.show()

# 3. Monthly AQI Trend
df["month"] = df["date"].dt.to_period("M").astype(str)

monthly_aqi = df.groupby("month")["india_aqi"].mean()

plt.figure(figsize=(14, 6))
plt.plot(monthly_aqi.index, monthly_aqi.values, marker="o")
plt.title("Monthly Average AQI Trend")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("graphs/03_monthly_aqi_trend.png",dpi = 300)
plt.show()

# 4. Correlation Heatmap
numeric_columns = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max",
    "india_aqi"
]

plt.figure(figsize=(10, 7))
correlation = df[numeric_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Pollutants and AQI Correlation")
plt.tight_layout()
plt.savefig("graphs/04_correlation_heatmap.png",dpi = 300)
plt.show()

# 5. PM2.5 vs AQI
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df,
    x="pm2_5_mean",
    y="india_aqi",
    alpha=0.4
)

plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5 Concentration")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("graphs/05_pm25_vs_aqi.png",dpi = 300)
plt.show()

print("\n===== EDA COMPLETED SUCCESSFULLY =====")
