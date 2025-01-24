from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import sqlite3
import math

app = FastAPI()

# Serve the static files directory for the frontend
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Paths to the models
infection_model_path = "./models/infection_model.pkl"
organ_model_path = "./models/organ_model.pkl"
sepsis_model_path = "./models/sepsis_model.pkl"

# Load models
with open(infection_model_path, "rb") as f:
    infection_model = pickle.load(f)
with open(organ_model_path, "rb") as f:
    organ_model = pickle.load(f)
with open(sepsis_model_path, "rb") as f:
    sepsis_model = pickle.load(f)

# Features used in models
infection_features = ['WBC', 'Glucose', 'Temp', 'HR', 'Resp']
organ_features = ['Creatinine', 'Bilirubin_total', 'BUN', 'FiO2', 'SBP', 'MAP']

# SQLite database setup
db_path = "./data/sepsis_predictions.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# Create the database table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    PatientID TEXT,
    WBC REAL, Glucose REAL, Temp REAL, HR REAL, Resp REAL,
    Creatinine REAL, Bilirubin_total REAL, BUN REAL, FiO2 REAL, SBP REAL, MAP REAL,
    prob_infection REAL, prob_organ_dysfunction REAL, prob_sepsis REAL
)
""")
conn.commit()

# Request payload model
class Features(BaseModel):
    PatientID: str
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
        # Prepare data for prediction
        infection_data = np.array([getattr(features, f) for f in infection_features]).reshape(1, -1)
        organ_data = np.array([getattr(features, f) for f in organ_features]).reshape(1, -1)
        sepsis_data = np.hstack([infection_data, organ_data])

        prob_infection = float(infection_model.predict_proba(infection_data)[0][1])  # Convert to float
        prob_organ_dysfunction = float(organ_model.predict_proba(organ_data)[0][1])  # Convert to float
        prob_sepsis = float(sepsis_model.predict_proba(sepsis_data)[0][1])  # Convert to float

        # Define thresholds
        infection_threshold = 0.3
        organ_dysfunction_threshold = 0.5
        sepsis_threshold = 0.5

        # Threshold-based classifications
        infection_flag = bool(prob_infection >= infection_threshold)  # Convert to bool
        organ_dysfunction_flag = bool(prob_organ_dysfunction >= organ_dysfunction_threshold)  # Convert to bool

        # Calculate sepsis probability based on combination logic
        if prob_infection < infection_threshold and prob_organ_dysfunction < organ_dysfunction_threshold:
            prob_sepsis_combination = 0.0  # No significant sepsis risk
        else:
            prob_sepsis_combination = prob_infection * prob_organ_dysfunction
            prob_sepsis_combination = 1 / (1 + math.exp(-10 * (prob_sepsis_combination - 0.5)))

        # Ensure `prob_sepsis_combination` is a float
        prob_sepsis_combination = float(prob_sepsis_combination)
        if organ_dysfunction_flag == True or infection_flag == True:
            sepsis_flag = "True"
        else:
            sepsis_flag = bool(prob_sepsis_combination > sepsis_threshold)  # Convert to bool

        # Save prediction to the database
        cursor.execute("""
                    INSERT INTO predictions (
                        PatientID, WBC, Glucose, Temp, HR, Resp, Creatinine, Bilirubin_total, BUN, FiO2, SBP, MAP,
                        prob_infection, prob_organ_dysfunction, prob_sepsis
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
            features.PatientID, features.WBC, features.Glucose, features.Temp, features.HR, features.Resp,
            features.Creatinine, features.Bilirubin_total, features.BUN, features.FiO2,
            features.SBP, features.MAP, prob_infection, prob_organ_dysfunction, prob_sepsis_combination
        ))
        conn.commit()

        # Return a JSON response
        return {
            "prob_infection": prob_infection,
            "prob_organ_dysfunction": prob_organ_dysfunction,
            #"prob_sepsis": prob_sepsis,
            "prob_sepsis": prob_sepsis_combination,
            "infection_flag": infection_flag,
            "organ_dysfunction_flag": organ_dysfunction_flag,
            "sepsis_flag": sepsis_flag
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get("/realtime-data/")
async def get_realtime_data():
    try:
        # Query the database to get sepsis prediction counts
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prob_sepsis > 0.5")
        sepsis_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prob_sepsis <= 0.5")
        non_sepsis_count = cursor.fetchone()[0]

        return JSONResponse(content={"sepsis": sepsis_count, "non_sepsis": non_sepsis_count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
@app.get("/records/")
async def get_records():
    try:
        cursor.execute("SELECT * FROM predictions")
        records = cursor.fetchall()
        return {"records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
