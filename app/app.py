from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "housing.csv")

df = pd.read_csv(data_path).dropna()
X = pd.get_dummies(df.drop("median_house_value", axis=1))
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

@app.route("/")
def home():
    return jsonify({"message": "California Housing Price Predictor", "roll_no": "2022BCS0206", "status": "running"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/metrics")
def metrics():
    return jsonify({"roll_no": "2022BCS0206", "dataset_size": len(X_train), "RMSE": round(rmse, 4), "R2": round(r2, 4)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
