"""
Trains the Logistic Regression model used by the Streamlit app,
using the CLEANED cardiovascular dataset, and saves the fitted
model + scaler to disk as .pkl files.

Run this once (or whenever you retrain) BEFORE launching app.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# ---- Path to your cleaned dataset ----
DATA_PATH = r"E:\SEM 5\ML\project\cardio_cleaned.csv"

# Features used for prediction (id and cardio excluded on purpose:
# id is just a row identifier, cardio is the target we're predicting)
FEATURES = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active"
]

df = pd.read_csv(DATA_PATH, sep=";")

X = df[FEATURES]
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model trained successfully. Test accuracy: {acc:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved model.pkl and scaler.pkl in the current folder.")
