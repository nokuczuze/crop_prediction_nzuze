import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("maine_merged_total_production_weather.csv")

# log-transform production
df["log_production"] = np.log1p(df["total_production"])

# one-hot encode counties
df_encoded = pd.get_dummies(df, columns=["county"], drop_first=True)

# only use county + year (drop weather because it seems theres not much variation in weather)
feature_cols = ["year"] + [col for col in df_encoded.columns if col.startswith("county_")]

X = df_encoded[feature_cols]
y = df_encoded["log_production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

test_index = X_test.index

comparison = pd.DataFrame({
    "year": df.loc[test_index, "year"].values,
    "county": df.loc[test_index, "county"].values,
    "actual_production": y_true,
    "predicted_production": y_pred
})

print("\nTest set predictions:")
print(comparison)


print("Baseline County-Year Model")
print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)

import matplotlib.pyplot as plt

plt.scatter(y_true, y_pred)
plt.xlabel("Actual production")
plt.ylabel("Predicted production")
plt.title("Actual vs predicted county production")
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()])
plt.tight_layout()
plt.show()


