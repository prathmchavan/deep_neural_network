from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os 

# Initialize FastAPI app
app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths to models (Change the model name for deep neural network)
rf_model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

# Load models
try:
    # Load that model below change the importing
    with open(rf_model_path, "rb") as rf_file:
        random_forest_model = pickle.load(rf_file)
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")
# Define input structure
class PredictionInput(BaseModel):
    feature_vector: list

# Routes
@app.post("/predict/dnn")
async def predict_decision_tree(data: PredictionInput):
    try:
        input_data = np.array(data.feature_vector).reshape(1, -1)
        prediction = decision_tree_model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the ML API! Use /predict/random-forest or /predict/decision-tree."}
