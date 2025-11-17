import mlflow
import os
from mlflow.tracking import MlflowClient
from app.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_PASSWORD
)

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if MLFLOW_USERNAME and MLFLOW_PASSWORD:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

    # Certificado autofirmado
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ["CURL_CA_BUNDLE"] = ""


def get_latest_model_version(model_name: str) -> int:
    """
    Obtiene la versión más reciente del modelo registrado en MLflow.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(f"No se encontraron versiones para el modelo '{model_name}'")

    # Ordenar por número de versión y tomar el más alto
    latest = max(versions, key=lambda v: int(v.version))

    print(f"🔎 Última versión encontrada para '{model_name}': {latest.version}")
    return int(latest.version)


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

    name = model_names_svr[zone] if svm else model_names_xgb[zone]

    # Obtener versión más reciente del modelo
    latest_version = get_latest_model_version(name)

    model_uri = f"models:/{name}/{latest_version}"
    print(f"🔄 Cargando modelo desde: {model_uri}")

    return mlflow.pyfunc.load_model(model_uri)
