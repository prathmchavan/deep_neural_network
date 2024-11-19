from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Base directory for relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to the deep neural network model
dnn_model_path = os.path.join(BASE_DIR, "models", "dnn_model.keras")

# Load the DNN model
try:
    dnn_model = load_model(dnn_model_path)
except Exception as e:
    raise RuntimeError(f"Error loading DNN model: {str(e)}")

# Define input structure
class PredictionInput(BaseModel):
    feature_vector: list

# Routes
@app.post("/predict/dnn")
async def predict_dnn(data: PredictionInput):
    try:
        # Convert input data to a NumPy array and reshape
        input_data = np.array(data.feature_vector).reshape(1, -1)
        
        # Make predictions using the DNN model
        prediction = dnn_model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the ML API! Use /predict/dnn to make predictions."}

@app.get("/github")
async def root():
    return {"message":"i am in github folder"}