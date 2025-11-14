import mlflow
import os
from app.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_PASSWORD,
    MODEL_VERSION
)

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if MLFLOW_USERNAME and MLFLOW_PASSWORD:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

    # Certificado autofirmado
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ["CURL_CA_BUNDLE"] = ""


def load_model_for_zone(zone: int, svm: bool):
    setup_mlflow()

    model_names_svr = {
        1: "svr_zone_1_power_consumption_power_consumption",
        2: "svr_zone_2__power_consumption_power_consumption",
        3: "svr_zone_3__power_consumption_power_consumption",
    }

    model_names_xgb = {
        1: "xgb_zone_1_power_consumption_power_consumption",
        2: "xgb_zone_2__power_consumption_power_consumption",
        3: "xgb_zone_3__power_consumption_power_consumption",
    }

    name = model_names_xgb[zone]

    if(svm):
        name = model_names_svr[zone]
        
    model_uri = f"models:/{name}/{MODEL_VERSION}"
    print(f"🔄 Cargando modelo desde: {model_uri}")

    return mlflow.pyfunc.load_model(model_uri)
