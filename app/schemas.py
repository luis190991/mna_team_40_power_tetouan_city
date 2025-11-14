from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    general_diffuse_flows: float
    diffuse_flows: float
    hour: float
    day: float
    month: float
    year: float
    zone: int = Field(..., ge=1, le=3, description="Zona del modelo a utilizar")
    svm: bool

class PredictionOutput(BaseModel):
    prediction: float
