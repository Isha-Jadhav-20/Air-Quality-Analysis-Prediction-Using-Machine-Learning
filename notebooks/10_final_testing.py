import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs("graphs", exist_ok=True)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


# 1. Load Cleaned Dataset
df = pd.read_csv("dataset/cleaned_aqi.csv")

df["date"] = pd.to_datetime(df["date"])

print("Dataset Shape:", df.shape)


# 2. Select Features and Target
features = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max",
    "city"
]

# 3. Chronological Train-Test Split
dates = sorted(df["date"].unique())

split_index = int(len(dates) * 0.8)
split_date = dates[split_index]

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]

X_test = test_df[features]
y_test = test_df["india_aqi"]

print("\nTesting Records:", len(test_df))
print("Split Date:", split_date)


# 4. Load Improved Model
model = joblib.load("models/improved_aqi_model.pkl")

print("\nImproved Model Loaded Successfully!")


# 5. Make Predictions
predictions = model.predict(X_test)

# AQI prediction should remain within the dataset's target range
predictions = np.clip(predictions, 0, 500)


# 6. Calculate Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n========== FINAL MODEL PERFORMANCE ==========")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 4))


# 7. Create Prediction Results
results = test_df[
    ["date", "city", "india_aqi"]
].copy()

results["predicted_aqi"] = predictions

results["prediction_error"] = (
    results["india_aqi"] - results["predicted_aqi"]
)

results["absolute_error"] = abs(
    results["prediction_error"]
)

results = results.rename(
    columns={"india_aqi": "actual_aqi"}
)

# 8. Display First 10 Predictions
print("\n========== SAMPLE PREDICTIONS ==========")
print(results.head(10).to_string(index=False))


# 9. High AQI Prediction Analysis
high_aqi = results[results["actual_aqi"] >= 400]

print("\n========== HIGH AQI ANALYSIS ==========")
print("High AQI Records:", len(high_aqi))

if len(high_aqi) > 0:
    print(
        "High AQI MAE:",
        round(high_aqi["absolute_error"].mean(), 3)
    )

    print("\nHigh AQI Prediction Samples:")
    print(
        high_aqi.sort_values(
            "absolute_error",
            ascending=False
        ).head(10).to_string(index=False)
    )


# 10. Save Prediction Results
os.makedirs("models", exist_ok=True)

results.to_csv(
    "models/final_aqi_predictions.csv",
    index=False
)

print("\nPrediction results saved successfully!")


# 11. Create Graph Folder
os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")


# Graph 1: Actual vs Predicted AQI
plt.figure(figsize=(8, 6))

plt.scatter(
    results["actual_aqi"],
    results["predicted_aqi"],
    alpha=0.4
)

plt.plot([0, 500], [0, 500], "r--")

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")

plt.savefig(
    "graphs/12_final_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Graph 2: Prediction Error Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    results["absolute_error"],
    bins=40,
    kde=True
)

plt.xlabel("Absolute Prediction Error")
plt.ylabel("Number of Records")
plt.title("Final Prediction Error Distribution")

plt.savefig(
    "graphs/13_final_prediction_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Graph 3: High AQI Actual vs Predicted
if len(high_aqi) > 0:

    plt.figure(figsize=(10, 5))

    plt.scatter(
        high_aqi["actual_aqi"],
        high_aqi["predicted_aqi"],
        alpha=0.6
    )

    plt.plot([400, 500], [400, 500], "r--")

    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("High AQI: Actual vs Predicted")

    plt.savefig(
        "graphs/14_high_aqi_prediction.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


print("\nPart 10 Completed Successfully!")
