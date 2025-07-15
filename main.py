# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained XGBoost models
model_shock = joblib.load("model_shock_absorber.pkl")
model_tire = joblib.load("model_tire_wear.pkl")
model_brake = joblib.load("model_brake_degradation.pkl")
model_hydraulic = joblib.load("model_hydraulic_failure.pkl")

# FastAPI instance
app = FastAPI()


# Define input schema
class LandingGearData(BaseModel):
    landing_cycles: int
    load_during_landing: float
    taxiing_duration: float
    speed_during_landing: float
    braking_force: float
    tire_pressure: float
    hydraulic_pressure: float
    runway_condition: str  # 'dry', 'wet', or 'snowy'


# Feature preprocessor
def preprocess_input(data: LandingGearData):
    base_features = [
        data.landing_cycles,
        data.load_during_landing,
        data.taxiing_duration,
        data.speed_during_landing,
        data.braking_force,
        data.tire_pressure,
        data.hydraulic_pressure,
    ]

    # One-hot encode 'runway_condition' (dry/wet only, snowy is dropped base)
    dry = 1 if data.runway_condition == "dry" else 0
    wet = 1 if data.runway_condition == "wet" else 0

    # Final feature vector
    features = base_features + [dry, wet]
    return np.array(features).reshape(1, -1)


# Prediction endpoint
@app.post("/predict")
def predict(data: LandingGearData):
    features = preprocess_input(data)

    return {
        "shock_absorber_failure": int(model_shock.predict(features)[0]),
        "tire_wear": int(model_tire.predict(features)[0]),
        "brake_degradation": int(model_brake.predict(features)[0]),
        "hydraulic_system_failure": int(model_hydraulic.predict(features)[0]),
    }
