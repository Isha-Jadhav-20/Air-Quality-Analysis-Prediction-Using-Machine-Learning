import pandas as pd
import numpy as np
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

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

X = df[features]
y = df["india_aqi"]


# 3. Chronological Train-Test Split
dates = sorted(df["date"].unique())

split_index = int(len(dates) * 0.8)
split_date = dates[split_index]

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

print("\nTraining Data:", train_df.shape)
print("Testing Data:", test_df.shape)
print("Split Date:", split_date)


# 4. Create Training and Validation Data
train_dates = sorted(train_df["date"].unique())

val_index = int(len(train_dates) * 0.8)
val_date = train_dates[val_index]

training_data = train_df[train_df["date"] < val_date]
validation_data = train_df[train_df["date"] >= val_date]

X_train = training_data[features]
y_train = training_data["india_aqi"]

X_val = validation_data[features]
y_val = validation_data["india_aqi"]

print("\nModel Training Rows:", len(X_train))
print("Validation Rows:", len(X_val))


# 5. Preprocessing
numeric_features = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max"
]

categorical_features = ["city"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False),
         categorical_features)
    ]
)


# 6. Try Different Models
models = {
    "Random Forest 1": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest 2": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest 3": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=2,
        subsample=0.8,
        random_state=42
    )
}


# 7. Train and Evaluate Models on Validation Data
results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("\nTraining:", name)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 4))


# 8. Compare Validation Results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values("RMSE")

print("\n========== VALIDATION RESULTS ==========")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]

print("\nBest Model:", best_model_name)


# 9. Train Best Model on Full Training Data
best_model = models[best_model_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(
    train_df[features],
    train_df["india_aqi"]
)


# 10. Final Evaluation on Test Data
X_test = test_df[features]
y_test = test_df["india_aqi"]

test_predictions = final_pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

print("\n========== FINAL TEST RESULTS ==========")
print("Selected Model:", best_model_name)
print("MAE:", round(test_mae, 3))
print("MSE:", round(test_mse, 3))
print("RMSE:", round(test_rmse, 3))
print("R2 Score:", round(test_r2, 4))


# 11. Save Final Model
os.makedirs("models", exist_ok=True)

joblib.dump(
    final_pipeline,
    "models/improved_aqi_model.pkl"
)

results_df.to_csv(
    "models/improvement_comparison.csv",
    index=False
)

print("\nImproved model saved successfully!")
