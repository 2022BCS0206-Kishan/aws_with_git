import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json

df = pd.read_csv("data/housing.csv").dropna()
X = pd.get_dummies(df.drop("median_house_value", axis=1))
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

print(f"Roll No     : 2022BCS0206")
print(f"Dataset size: {len(X_train)} training samples")
print(f"RMSE        : {rmse:.4f}")
print(f"R²          : {r2:.4f}")

metrics = {
    "roll_no": "2022BCS0206",
    "dataset_size": len(X_train),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
