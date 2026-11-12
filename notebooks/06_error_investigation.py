
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("dataset/cleaned_aqi.csv")
df["date"] = pd.to_datetime(df["date"])

# Features
numeric_features = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max"
]

X = df[numeric_features + ["city"]]
y = df["india_aqi"]

# Same chronological split
unique_dates = sorted(df["date"].unique())
split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# Load model
model = joblib.load("models/best_aqi_model.pkl")

predictions = model.predict(X_test)

# Prediction analysis
results = df.loc[test_mask, ["date", "city"]].copy()

results["Actual AQI"] = y_test.values
results["Predicted AQI"] = predictions
results["Absolute Error"] = np.abs(
    y_test.values - predictions
)

# 1. Training and testing target ranges
print("\n===== AQI RANGE =====")
print("Training minimum:", y_train.min())
print("Training maximum:", y_train.max())
print("Testing minimum:", y_test.min())
print("Testing maximum:", y_test.max())

# 2. Extreme AQI records
print("\n===== RECORDS WITH AQI >= 400 =====")

extreme = results[results["Actual AQI"] >= 400]

print(extreme.sort_values(
    "Actual AQI", ascending=False
).head(20).to_string(index=False))

# 3. City-wise evaluation
results["Absolute Error"] = results["Absolute Error"]

print("\n===== CITY-WISE MAE =====")

print(
    results.groupby("city")
    .apply(
        lambda group: mean_absolute_error(
            group["Actual AQI"],
            group["Predicted AQI"]
        ),
        include_groups=False
    )
    .sort_values(ascending=False)
    .head(10)
)

# 4. AQI range-wise performance
results["AQI Range"] = pd.cut(
    results["Actual AQI"],
    bins=[0, 100, 200, 300, 400, 500],
    labels=["0-100", "101-200", "201-300", "301-400", "401-500"],
    include_lowest=True
)

print("\n===== PERFORMANCE BY AQI RANGE =====")

print(
    results.groupby("AQI Range", observed=False)
    .agg(
        Records=("Actual AQI", "count"),
        MAE=("Absolute Error", "mean")
    )
)

# 5. Check pollutant values for extreme records
print("\n===== EXTREME RECORD POLLUTANTS =====")

extreme_indices = results[
    results["Actual AQI"] >= 400
].index

print(
    df.loc[
        extreme_indices,
        ["city"] + numeric_features + ["india_aqi"]
    ].head(20).to_string(index=False)
)

# 6. Inspect pollutants for largest prediction errors

results["Absolute Error"] = np.abs(
    results["Actual AQI"] - results["Predicted AQI"]
)

largest_errors = results.sort_values(
    by="Absolute Error",
    ascending=False
).head(15)

print("\n===== LARGEST ERROR RECORDS WITH POLLUTANTS =====")

error_details = largest_errors.merge(
    df[["date", "city",
        "pm2_5_mean", "pm10_mean",
        "no2_mean", "so2_mean",
        "o3_8h_max", "co_8h_max"]],
    on=["date", "city"],
    how="left"
)

print(error_details.to_string(index=False))
