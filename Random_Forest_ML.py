import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load merged dataset
df = pd.read_csv("maine_merged_total_production_weather.csv")

# 2. Choose features and target
feature_cols = [
    "temp_lo_F",
    "temp_hi_F",
    "temp_avg_F",
    "total_precip_inches",
    "total_precip_days",
    "year"          
]

target_col = "total_production"


df_model = df[feature_cols + [target_col]].dropna()

X = df_model[feature_cols]
y = df_model[target_col]

# 3. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Define and train Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# 5. Evaluate model
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Random Forest results")
print("---------------------")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2:  {r2:.3f}")






