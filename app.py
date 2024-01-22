from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

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


@app.get("/")
async def read_root():
    return "Welcome!"


@app.post("/predict")
async def predict(input_data: InputData):
    try:

        input_data = input_data.dict()

        # Convert input data to DataFrame
        input_data = pd.DataFrame([input_data])

        # Load the model pipeline
        pipeline = joblib.load("model/inference_pipeline.pkl")
        # Make predictions
        prediction = pipeline.predict(input_data)
        # Assuming prediction is a single value
        result = {"prediction": prediction[0]}

        return result

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error during prediction: {str(e)}")
