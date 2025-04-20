from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
# Load the trained model
model = joblib.load("random_forest.pkl")

# Define the input data structure (features)
class PredictionRequest(BaseModel):
    Ped_South: float
    Ped_North: float
    Bike_North: float
    Bike_South: float
    timestamp: datetime



# FastAPI instance
app = FastAPI()

import pandas as pd

# Define the inference endpoint
@app.post("/predict/")
def predict(request: PredictionRequest):
     # Simulate a mini dataframe for feature calculations
    total_traffic = request.Ped_South + request.Ped_North + request.Bike_South + request.Bike_North

     # Step 2: Extract time features
    hour = request.timestamp.hour
    dayofweek = request.timestamp.weekday()  # Monday=0, Sunday=6


    df = pd.DataFrame([{
        "BGT_North_of_NE_70th_Total": total_traffic,
        "Ped_South": request.Ped_South,
        "Ped_North": request.Ped_North,
        "Bike_North": request.Bike_North,
        "Bike_South": request.Bike_South,
        "hour": hour,
        "dayofweek": dayofweek
    }])

    # Feature Engineering — customize this logic as needed
    df["Weekdays"] = 1 if dayofweek < 5 else 0 # You could dynamically pass date in the future
    df["Ped_South_Rolling_Avg"] = df["Ped_South"]  # Placeholder logic
    df["Bike_South_Rolling_Avg"] = df["Bike_South"]
    df["Ped_North_Rolling_Avg"] = df["Ped_North"]
    df["Bike_North_Rolling_Avg"] = df["Bike_North"]
    
    # Simple z-score estimates — use rolling means/std in real code
    df["Ped_North_Z"] = 0  # Placeholder — ideally from real-time baseline
    df["Ped_South_Z"] = 0
    df["Bike_North_Z"] = 0
    df["Bike_South_Z"] = 0

    df["anamoly"] = 1 if total_traffic >= 500 else 0
    df["lag_1hr"] = df["BGT_North_of_NE_70th_Total"]  # Use real lag history if available
    df["lag_2hr"] = df["BGT_North_of_NE_70th_Total"]
    df["lag_3hr"] = df["BGT_North_of_NE_70th_Total"]

    # Make prediction
    input_data = df.values
    prediction = model.predict(input_data)
    prediction_result = int(prediction[0])

    return {"target_3hr_prediction": prediction_result}