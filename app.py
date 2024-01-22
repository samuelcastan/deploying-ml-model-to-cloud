from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()


class InputData(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str

# Load the scikit-learn pipeline from a pickle file
pipeline = joblib.load("model/inference_pipeline.pkl")


@app.get("/")
def read_root():
    return "Hello World"

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input data to a numpy array for prediction
        input_array = np.array([[
            input_data.attribute1,
            input_data.attribute2,
            input_data.attribute3
        ]])

        # Perform prediction using the loaded pipeline
        prediction = pipeline.predict(input_array)

        # Assuming a regression model, modify as needed for classification, etc.
        result = {"prediction": float(prediction[0])}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Run your FastAPI application using uvicorn:
# uvicorn your_script_name:app --reload