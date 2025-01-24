from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Serve the static directory
app.mount("/static", StaticFiles(directory="api/static"), name="static")

infection_model_path = "./models/infection_model.pkl"
organ_model_path = "./models/organ_model.pkl"

# Load models
with open(infection_model_path, "rb") as f:
    infection_model = pickle.load(f)
with open(organ_model_path, "rb") as f:
    organ_model = pickle.load(f)

infection_features = ['WBC', 'Glucose', 'Temp', 'HR', 'Resp']
organ_features = ['Creatinine', 'Bilirubin_total', 'BUN', 'FiO2', 'SBP', 'MAP']

class Features(BaseModel):
    WBC: float
    Glucose: float
    Temp: float
    HR: float
    Resp: float
    Creatinine: float
    Bilirubin_total: float
    BUN: float
    FiO2: float
    SBP: float
    MAP: float

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("api/static/index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/predict/")
async def predict(features: Features):
    try:
        infection_data = np.array([getattr(features, f) for f in infection_features]).reshape(1, -1)
        organ_data = np.array([getattr(features, f) for f in organ_features]).reshape(1, -1)

        prob_infection = infection_model.predict_proba(infection_data)[0][1]
        prob_organ_dysfunction = organ_model.predict_proba(organ_data)[0][1]
        prob_sepsis = prob_infection * prob_organ_dysfunction

        return {
            "prob_infection": prob_infection,
            "prob_organ_dysfunction": prob_organ_dysfunction,
            "prob_sepsis": prob_sepsis,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
