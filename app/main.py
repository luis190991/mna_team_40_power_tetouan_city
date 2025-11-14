from fastapi import FastAPI, HTTPException
from app.schemas import PredictionInput, PredictionOutput
from app.loader import load_model_for_zone
import numpy as np
import pandas as pd

app = FastAPI(
    title="Power Consumption Model API",
    description="API para modelos MLflow (SVR/XGB) por zona",
    version="1.0.0"
)

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(payload: PredictionInput):
    try:
        # 1. Cargar modelo
        model = load_model_for_zone(payload.zone, payload.svm)

        # 2. Convertir a array en el orden correcto
        df = pd.DataFrame([{
            "Temperature": payload.Temperature,
            "Humidity": payload.Humidity,
            "Wind Speed": payload.Wind_Speed,
            "general diffuse flows": payload.general_diffuse_flows,
            "diffuse flows": payload.diffuse_flows,
            "hour": payload.hour,
            "day": payload.day,
            "month": payload.month,
            "year": payload.year
        }])

        # 3. Predicción
        prediction = model.predict(df)

        return PredictionOutput(prediction=float(prediction[0]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
