# ======================================================
# Flood Prediction System
# File : train_model.py
# Author : Your Name
# ======================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------
# Folder Paths
# ------------------------------------------------------

DATASET_PATH = "dataset/flood_data.csv"
MODEL_FOLDER = "model"

MODEL_FILE = os.path.join(MODEL_FOLDER, "flood_model.pkl")
SCALER_FILE = os.path.join(MODEL_FOLDER, "scaler.pkl")

# ------------------------------------------------------
# Create model folder if it doesn't exist
# ------------------------------------------------------

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# ------------------------------------------------------
# Load Dataset
# ------------------------------------------------------

print("Loading Dataset...")

df = pd.read_csv(DATASET_PATH)

print(df.head())

# ------------------------------------------------------
# Dataset Columns
# ------------------------------------------------------
# rainfall
# temperature
# humidity
# water_level
# soil_moisture
# flood
# ------------------------------------------------------

print("\nDataset Shape:", df.shape)

# ------------------------------------------------------
# Check Missing Values
# ------------------------------------------------------

print("\nMissing Values")

print(df.isnull().sum())

# ------------------------------------------------------
# Remove Missing Values
# ------------------------------------------------------

df = df.dropna()

# ------------------------------------------------------
# Features
# ------------------------------------------------------

X = df[[
    "rainfall",
    "temperature",
    "humidity",
    "water_level",
    "soil_moisture"
]]

# ------------------------------------------------------
# Target
# ------------------------------------------------------

y = df["flood"]

# ------------------------------------------------------
# Train Test Split
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------
# Feature Scaling
# ------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------
# Model
# ------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ------------------------------------------------------
# Prediction
# ------------------------------------------------------

predictions = model.predict(X_test_scaled)

# ------------------------------------------------------
# Accuracy
# ------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("Model Accuracy")
print("==============================")

print(round(accuracy * 100, 2), "%")

print("\nClassification Report")

print(classification_report(y_test, predictions))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, predictions))

# ------------------------------------------------------
# Save Model
# ------------------------------------------------------

joblib.dump(model, MODEL_FILE)

joblib.dump(scaler, SCALER_FILE)

print("\nModel Saved Successfully")

print("Location :", MODEL_FILE)

print("Scaler Saved Successfully")

print("Location :", SCALER_FILE)

print("\nTraining Completed.")