# ==========================================================
# Flood Prediction System
# File : flood_prediction.py
# Description : Load trained model and predict flood risk
# ==========================================================

import os
import joblib
import numpy as np

# ----------------------------------------------------------
# Model Paths
# ----------------------------------------------------------

MODEL_PATH = "model/flood_model.pkl"
SCALER_PATH = "model/scaler.pkl"

# ----------------------------------------------------------
# Load Model and Scaler
# ----------------------------------------------------------

model = None
scaler = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✓ Flood Prediction Model Loaded")
else:
    print("✗ flood_model.pkl not found")

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print("✓ Scaler Loaded")
else:
    print("✗ scaler.pkl not found")


# ----------------------------------------------------------
# Prediction Function
# ----------------------------------------------------------

def predict_flood(
    rainfall,
    temperature,
    humidity,
    water_level,
    soil_moisture
):
    """
    Predict flood risk.

    Parameters
    ----------
    rainfall : float
    temperature : float
    humidity : float
    water_level : float
    soil_moisture : float

    Returns
    -------
    str
        Flood Risk
    """

    data = np.array([[
        rainfall,
        temperature,
        humidity,
        water_level,
        soil_moisture
    ]])

    # Scale features if scaler exists
    if scaler is not None:
        data = scaler.transform(data)

    # Use trained model if available
    if model is not None:
        prediction = model.predict(data)[0]
        return str(prediction)

    # ------------------------------------------------------
    # Fallback Rule-Based Prediction
    # ------------------------------------------------------

    if rainfall >= 250 or water_level >= 8:
        return "High Risk"

    elif rainfall >= 180 or water_level >= 6:
        return "Medium Risk"

    else:
        return "Low Risk"


# ----------------------------------------------------------
# Command Line Testing
# ----------------------------------------------------------

if __name__ == "__main__":

    print("\nFlood Prediction Test")
    print("-" * 35)

    rainfall = float(input("Rainfall (mm): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    water_level = float(input("Water Level (m): "))
    soil = float(input("Soil Moisture (%): "))

    result = predict_flood(
        rainfall,
        temperature,
        humidity,
        water_level,
        soil
    )

    print("\nPredicted Flood Risk:", result)