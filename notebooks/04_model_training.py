import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_aqi.csv")
df["date"] = pd.to_datetime(df["date"])

print("Dataset Shape:", df.shape)

# 2. Select input features and target
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

# 3. Chronological train-test split
unique_dates = sorted(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Test Start Date:", split_date)

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("city", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_features)
    ]
)

# 5. Define ML models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
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

results = []
trained_models = {}

# 6. Train and evaluate models
for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

# 7. Compare model performance
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# 8. Select best model based on lowest RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_aqi_model.pkl")

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nBest Model:", best_model_name)
print("Best model saved successfully!")
print("Model comparison saved successfully!")
