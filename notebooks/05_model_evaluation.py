import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("graphs", exist_ok=True)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1) Load dataset
df = pd.read_csv("dataset/cleaned_aqi.csv")
df["date"] = pd.to_datetime(df["date"])

numeric_features = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max"
]

categorical_features = ["city"]

X = df[numeric_features + categorical_features]
y = df["india_aqi"]

# 2) Same chronological split as training
unique_dates = sorted(df["date"].unique())
split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

test_mask = df["date"] >= split_date

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

# 3) Load trained model
model = joblib.load("models/best_aqi_model.pkl")

# 4) Predict AQI
y_pred = model.predict(X_test)

# 5) Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n----- RANDOM FOREST MODEL EVALUATION ------")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# 6) Feature importance
preprocessor = model.named_steps["preprocessing"]
random_forest = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importance = random_forest.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Group city one-hot features together
importance_df["Feature Group"] = importance_df["Feature"].apply(
    lambda x: "City" if x.startswith("city__") else x.replace("numeric__", "")
)

grouped_importance = (
    importance_df.groupby("Feature Group")["Importance"]
    .sum()
    .sort_values(ascending=False)
)

print("\n----- FEATURE IMPORTANCE -----")
print(grouped_importance)

# Feature importance graph
plt.figure(figsize=(10, 6))
grouped_importance.sort_values().plot(kind="barh")
plt.title("AQI Prediction - Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("graphs/06_Feature_importance_graph.png", dpi=300)
plt.show()

# 7) Actual vs Predicted AQI
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.savefig("graphs/07_Actual_vs_Predicted_AQI.png", dpi=300)
plt.show()

# 8) Prediction error distribution
errors = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=40, kde=True)
plt.title("Prediction Error Distribution")
plt.xlabel("Actual AQI - Predicted AQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graphs/08_Prediction_error_distribution.png", dpi=300)
plt.show()

# 9) Save sample predictions
prediction_df = pd.DataFrame({
    "Actual AQI": y_test.values,
    "Predicted AQI": np.round(y_pred, 2),
    "Error": np.round(errors.values, 2)
})

prediction_df.to_csv(
    "models/aqi_predictions.csv",
    index=False
)

print("\nPrediction results saved successfully!")
print("\n----- SAMPLE PREDICTIONS -----")
print(prediction_df.head(10))

# 10) Analyze largest prediction errors

error_analysis = df.loc[
    test_mask, ["date", "city"]
].copy()

error_analysis["Actual AQI"] = y_test.values
error_analysis["Predicted AQI"] = np.round(y_pred, 2)
error_analysis["Absolute Error"] = np.abs(
    y_test.values - y_pred
)

print("\n----- ERROR STATISTICS -----")
print(error_analysis["Absolute Error"].describe(
    percentiles=[0.50, 0.90, 0.95, 0.99]
))

print("\n----- TOP 10 LARGEST PREDICTION ERRORS -----")
print(
    error_analysis.sort_values(
        by="Absolute Error",
        ascending=False
    ).head(10).to_string(index=False)
)