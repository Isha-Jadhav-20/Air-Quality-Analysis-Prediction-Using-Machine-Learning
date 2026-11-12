
import pandas as pd

# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_aqi.csv")

df["date"] = pd.to_datetime(df["date"])

print("Dataset Shape:", df.shape)

# 2. Split data into training and testing
unique_dates = sorted(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]

print("\nTraining Records:", len(train_df))
print("Testing Records:", len(test_df))
print("Split Date:", split_date)

# 3. Select columns for analysis
columns = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max",
    "india_aqi"
]

# 4. Compare city-wise data
cities = ["Jaipur", "Jodhpur", "Delhi", "Agra"]

for city in cities:

    print("\n================================")
    print("CITY:", city)
    print("================================")

    city_train = train_df[train_df["city"] == city]
    city_test = test_df[test_df["city"] == city]

    print("\nTRAINING DATA")
    print("Total Records:", len(city_train))
    print(city_train[columns].describe().round(2))

    print("\nTESTING DATA")
    print("Total Records:", len(city_test))
    print(city_test[columns].describe().round(2))

# 5. Check monthly AQI trend in testing data
test_df = test_df.copy()

test_df["month"] = test_df["date"].dt.to_period("M")

monthly_aqi = test_df.groupby(
    ["city", "month"]
)["india_aqi"].mean().reset_index()

print("\n===== MONTHLY AQI COMPARISON =====")

selected_cities = monthly_aqi[
    monthly_aqi["city"].isin(cities)
]

print(selected_cities.to_string(index=False))

print("\n===== DATA DISTRIBUTION CHECK COMPLETED =====")
