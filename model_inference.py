from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest.pkl")

# Define the input data structure (features)
class PredictionRequest(BaseModel):
    BGT_North_of_NE_70th_Total: float
    Ped_South: float
    Ped_North: float
    Bike_North: float
    Bike_South: float
    Weekdays: int
    Ped_South_Rolling_Avg: float
    Bike_South_Rolling_Avg: float
    Ped_North_Rolling_Avg: float
    Bike_North_Rolling_Avg: float
    Ped_North_Z: float
    Ped_South_Z: float
    Bike_North_Z: float
    Bike_South_Z: float
    anamoly: int
    lag_1hr: float
    lag_2hr: float
    lag_3hr: float


# FastAPI instance
app = FastAPI()

# Define the inference endpoint
@app.post("/predict/")
def predict(request: PredictionRequest):
    # Convert input data into numpy array (features for prediction)
    input_data = np.array([[
        request.BGT_North_of_NE_70th_Total,
        request.Ped_South,
        request.Ped_North,
        request.Bike_North,
        request.Bike_South,
        request.Weekdays,
        request.Ped_South_Rolling_Avg,
        request.Bike_South_Rolling_Avg,
        request.Ped_North_Rolling_Avg,
        request.Bike_North_Rolling_Avg,
        request.Ped_North_Z,
        request.Ped_South_Z,
        request.Bike_North_Z,
        request.Bike_South_Z,
        request.anamoly,
        request.lag_1hr,
        request.lag_2hr,
        request.lag_3hr
    ]])

    # Perform prediction
    prediction = model.predict(input_data)

    # Convert the prediction to native Python types (int or float)
    prediction_result = int(prediction[0])  # Convert numpy.int64 to int

    # Return the prediction
    return {"target_3hr_prediction": prediction_result}
