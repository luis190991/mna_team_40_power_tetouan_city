import os
import mlflow
from mlflow import log_metric, log_param, log_artifact
from dotenv import load_dotenv
from datetime import datetime
import logging




class MLflowLogger:
    """
    Clase para gestionar el registro de experimentos en MLflow remoto,
    compatible con autenticación Nginx (Basic Auth o Bearer Token).
    """

    def __init__(self, env_path="mlflow.env"):
        # Logging local
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

        # 1️⃣ Cargar variables del entorno
        if os.path.exists(env_path):
            load_dotenv(env_path)
            self.logger.info("✅ Archivo .env cargado correctamente.")
        else:
            self.logger.warning("⚠️ No se encontró archivo .env, usando variables del entorno.")

        # 2️⃣ Configurar variables de conexión
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.username = os.getenv("MLFLOW_TRACKING_USERNAME")
        self.password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        self.token = os.getenv("MLFLOW_TRACKING_TOKEN")
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")

        # 3️⃣ ⚙️ Desactivar verificación SSL si el entorno lo indica
        if os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "false").lower() == "true":
            import urllib3, warnings
            from urllib3.exceptions import InsecureRequestWarning
            urllib3.disable_warnings(InsecureRequestWarning)
            os.environ["CURL_CA_BUNDLE"] = ""  # ← clave: desactiva validación TLS
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
            self.logger.warning("⚠️ SSL verification disabled for self-signed certificate")

        # 4️⃣ Configurar autenticación y conexión MLflow
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri or ""
        if self.username and self.password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password
        if self.token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = self.token

        # 5️⃣ Inicializar MLflow remoto
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self.logger.info(f"📡 Conectado a MLflow remoto: {self.tracking_uri}")
        self.logger.info(f"🧪 Experimento activo: {self.experiment_name}")

    # ----------------------------------------------------------
    def start_run(self, run_name=None, tags=None):
        """Inicia un nuevo experimento MLflow."""
        if run_name is None:
            run_name = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        self.logger.info(f"🚀 Iniciando experimento: {run_name}")

    # ----------------------------------------------------------
    def log_params(self, params: dict):
        """Registra parámetros del modelo o dataset."""
        for k, v in params.items():
            mlflow.log_param(k, v)
        self.logger.info(f"📘 Parámetros registrados: {list(params.keys())}")

    # ----------------------------------------------------------
    def log_metrics(self, metrics: dict):
        """Registra métricas (MAE, RMSE, R², etc.)."""
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        self.logger.info(f"📊 Métricas registradas: {metrics}")
    
    def log_metric(self, name, value):
        mlflow.log_metric(name, value)

    # ----------------------------------------------------------
    def log_artifact(self, filepath: str):
        """Sube artefactos (modelos, reportes, etc.) al servidor MLflow."""
        if os.path.exists(filepath):
            mlflow.log_artifact(filepath)
            self.logger.info(f"💾 Artefacto subido: {filepath}")
        else:
            self.logger.warning(f"⚠️ No se encontró el artefacto: {filepath}")

    # ----------------------------------------------------------
    def set_tags(self, tags: dict):
        """
        Asigna etiquetas (tags) a un run activo.
        Ejemplo:
        {"author": "Luis Ramirez", "framework": "scikit-learn"}
        """
        mlflow.set_tags(tags)
        self.logger.info(f"🏷️ Tags asignados: {tags}")

    def set_description(self, text: str):
        """
        Asigna una descripción visible en la interfaz web de MLflow (Overview).
        """
        mlflow.set_tag("mlflow.note.content", text)
        self.logger.info("📝 Descripción del experimento registrada.")
    
    def end_run(self):
        """Cierra la sesión actual."""
        mlflow.end_run()
        self.logger.info("🏁 Ejecución finalizada y registrada en MLflow remoto.")
