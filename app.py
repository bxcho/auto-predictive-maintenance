from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the FastAPI application
app = FastAPI(
    title="Automotive Predictive Maintenance API",
    description="Real-time equipment failure prediction using XGBoost.",
    version="1.0.0"
)

# Load the trained model into memory on startup
model = joblib.load("xgboost_model.pkl")

# Define the expected data structure for incoming requests
class SensorData(BaseModel):
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running. Send POST requests to /predict."}

@app.post("/predict")
def predict_failure(data: SensorData):
    # Convert incoming JSON data into a Pandas DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Get the raw probability of failure (Class 1)
    probability = float(model.predict_proba(input_data)[0][1])

    # CUSTOM THRESHOLD: Trigger alarm if probability is over 40%
    custom_threshold = 0.40
    is_failing = probability >= custom_threshold

    # Format the response
    return {
        "failure_prediction": 1 if is_failing else 0,
        "failure_probability": round(probability, 4),
        "system_status": "CRITICAL: Maintenance Required" if is_failing else "Healthy"
    }