import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load data
print("Loading data...")
df = pd.read_csv("employee_data.csv")

# Encode categoricals
le = LabelEncoder()
for col in ["dept", "education", "gender"]:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("promoted", axis=1)
y = df["promoted"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to model/model.pkl")
