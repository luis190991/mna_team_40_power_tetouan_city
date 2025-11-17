import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tools.logger import get_logger
import json
from tools.mlflow_logger import MLflowLogger
import subprocess
import mlflow


class TrainingPhase:
    """
    Fase de entrenamiento de modelos para predicción de consumo eléctrico
    (Tetouan City Power Consumption dataset).

    Entrena y evalúa modelos XGBoost y SVR por zona,
    guarda métricas y mejores modelos.
    """

    def __init__(self, processed_dir, model_dir):
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Datasets
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None

        # Resultados
        self.best_models = {"xgb": {}, "svr": {}}
        self.metrics = {"xgb": {}, "svr": {}}

    def load_data(self):
        """Carga los datasets procesados generados en la fase de preprocesamiento."""
        logger.info(f"Cargando datasets desde: {self.processed_dir}")

        self.X_train = pd.read_csv(os.path.join(self.processed_dir, 'X_train.csv'))
        self.X_val = pd.read_csv(os.path.join(self.processed_dir, 'X_val.csv'))
        self.X_test = pd.read_csv(os.path.join(self.processed_dir, 'X_test.csv'))

        self.Y_train = pd.read_csv(os.path.join(self.processed_dir, 'Y_train.csv'))
        self.Y_val = pd.read_csv(os.path.join(self.processed_dir, 'Y_val.csv'))
        self.Y_test = pd.read_csv(os.path.join(self.processed_dir, 'Y_test.csv'))

        logger.info(f"Datos cargados: X_train={self.X_train.shape}, Y_train={self.Y_train.shape}")

    def evaluate_model(self, model_name, y_true, y_pred):
        """Evalúa las métricas para cada zona."""
        results = {}
        for i, zone in enumerate(y_true.columns):
            y_true_col = y_true.iloc[:, i]
            y_pred_col = y_pred[:, i] if isinstance(y_pred, np.ndarray) else y_pred[zone]

            mae = mean_absolute_error(y_true_col, y_pred_col)
            mse = mean_squared_error(y_true_col, y_pred_col)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_col, y_pred_col)

            results[zone] = {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
            }

            logger.info(f"\n{model_name} - {zone}")
            logger.info(f"   MAE:  {mae:.4f}")
            logger.info(f"   RMSE: {rmse:.4f}")
            logger.info(f"   R2:   {r2:.4f}")
        return results

    # ----------------------------------------------------------
    def train_xgboost(self):
        """Entrena un modelo XGBoost por cada zona con GridSearchCV."""
        logger.info("Entrenando modelo XGBoost por zona...")

        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8]
        }

        y_pred_dict = {}

        for zone in self.Y_train.columns:
            logger.info(f"Ajustando XGBoost para {zone}...")

            xgb_base = XGBRegressor(
                random_state=42,
                objective='reg:squarederror',
                verbosity=0
            )

            y = self.Y_train[zone]
            grid = GridSearchCV(
                xgb_base, xgb_param_grid, cv=3,
                scoring='r2', n_jobs=-1, verbose=0
            )
            grid.fit(self.X_train, y)

            best_model = grid.best_estimator_
            self.best_models["xgb"][zone] = best_model

            y_pred = best_model.predict(self.X_test)
            y_pred_dict[zone] = y_pred

            logger.info(f"Mejor modelo XGB para {zone}: {grid.best_params_}")

        y_pred_xgb = pd.DataFrame(y_pred_dict, index=self.X_test.index)
        self.metrics["xgb"] = self.evaluate_model("XGBoost", self.Y_test, y_pred_xgb)

    # ----------------------------------------------------------
    def train_svr(self):
        """Entrena un modelo SVR por cada zona con GridSearchCV."""
        logger.info("Entrenando modelo SVR por zona...")

        svr_param_grid = {
            'C': [1, 10],
            'epsilon': [0.01, 0.1],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto']
        }

        y_pred_dict = {}

        for zone in self.Y_train.columns:
            logger.info(f"Ajustando SVR para {zone}...")

            y = self.Y_train[zone]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]

            svr_base = SVR()
            grid = GridSearchCV(
                svr_base, svr_param_grid, cv=3,
                scoring='r2', n_jobs=-1, verbose=0
            )
            grid.fit(self.X_train, y)

            best_model = grid.best_estimator_
            self.best_models["svr"][zone] = best_model

            y_pred = best_model.predict(self.X_test)
            y_pred_dict[zone] = y_pred

            logger.info(f"Mejor modelo SVR para {zone}: {grid.best_params_}")

        y_pred_svr = pd.DataFrame(y_pred_dict, index=self.X_test.index)
        self.metrics["svr"] = self.evaluate_model("SVR", self.Y_test, y_pred_svr)

    # ----------------------------------------------------------
    def save_artifacts(self):
        """Guarda modelos y métricas localmente."""
        logger.info("Guardando modelos y métricas...")

        for model_type, models in self.best_models.items():
            model_dir = os.path.join(self.model_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)

            for zone, model in models.items():
                model_name = f"{model_type}_{zone.replace(' ', '_').lower()}_power.pkl"
                path = os.path.join(self.model_dir, model_type, model_name)
                joblib.dump(model, path)
                logger.info(f"💾 Modelo guardado: {path}")

        metrics_path = os.path.join(self.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"📊 Métricas guardadas en {metrics_path}")



    # ----------------------------------------------------------

    def get_dvc_dataset_version(self):
        """
        Devuelve el hash o versión actual del dataset gestionado por DVC.
        Si no hay control DVC, devuelve 'unknown'.
        """
        try:
            # Ejecuta un comando DVC para obtener la última revisión del dataset
            result = subprocess.run(
                ["dvc", "status", "-c"],
                capture_output=True,
                text=True,
                check=False
            )

            # También puedes usar dvc get o dvc list para obtener hash del lock
            dvc_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False
            ).stdout.strip()

            if result.returncode == 0:
                return dvc_hash  # usa commit de git como referencia
            else:
                return f"unknown (DVC not synced, commit {dvc_hash})"

        except Exception as e:
            return f"error: {e}"

    def run_pipeline(self):
        """Ejecuta toda la fase de entrenamiento, registra métricas y modelos con MLflow."""
        import re
        import mlflow
        from mlflow import sklearn

        # Inicializa logger de MLflow
        ml_logger = MLflowLogger()
        ml_logger.start_run(run_name="Training_Tetouan_Model")

        # 1️⃣ Cargar datos
        self.load_data()

        # 2️⃣ Entrenar modelos
        self.train_xgboost()
        self.train_svr()

        # 3️⃣ Guardar artefactos localmente
        self.save_artifacts()

        # 4️⃣ Registrar parámetros del dataset
        ml_logger.log_params({
            "dataset_version": self.get_dvc_dataset_version(),
            "xgb_estimators": 200,
            "xgb_learning_rate": 0.1,
            "svr_kernels": "rbf",
            "cv_folds": 3
        })

        # 5️⃣ Registrar métricas (por modelo y zona)
        for model_type, results in self.metrics.items():
            for zone, metrics in results.items():
                for metric_name, value in metrics.items():
                    ml_logger.log_metric(f"{model_type}_{zone}_{metric_name}", value)

        # 6️⃣ Subir archivos clave como artefactos
        try:
            ml_logger.log_artifact("dvc.yaml")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo subir dvc.yaml: {e}")

        try:
            ml_logger.log_artifact("models/metrics.json")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo subir metrics.json: {e}")

        # 7️⃣ Registrar modelos en MLflow Model Registry
        for model_type, models in self.best_models.items():
            for zone, model in models.items():
                # Generar nombre válido
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{model_type}_{zone}_power_consumption").lower()
                local_model_path = os.path.join(
                    self.model_dir,
                    model_type,
                    f"{model_type}_{zone.replace(' ', '_').lower()}_power.pkl"
                )

                # Registrar modelo
                try:
                    input_example = self.X_test.iloc[:1]

                    # 1️⃣ Log del modelo (artefacto)
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=safe_name,    # <- reemplaza name=
                        input_example=input_example
                    )

                    run_id = mlflow.active_run().info.run_id
                    model_uri = f"runs:/{run_id}/{safe_name}"

                    # 2️⃣ Registrar en el Model Registry
                    mlflow.register_model(
                        model_uri=model_uri,
                        name=safe_name
                    )

                    logger.info(f"✅ Modelo registrado en MLflow: {safe_name}")

                except Exception as e:
                    logger.error(f"⚠️ Error registrando {safe_name}: {e}")


        # 8️⃣ Agregar etiquetas de contexto (metadatos)
        ml_logger.set_tags({
            "author": "Equipo 4",
            "env": "MLOps",
            "framework": "scikit-learn + XGBoost",
            "dataset": "Tetouan City Power Consumption",
            "description": "Entrenamiento y evaluación de modelos XGBoost y SVR por zona con trazabilidad completa (DVC + MLflow)."
        })

        # 9️⃣ Finalizar experimento
        ml_logger.end_run()

        logger.info("🏁 Fase de entrenamiento completada y registrada exitosamente en MLflow remoto.")




if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    MODEL_DIR = "models"

    logger = get_logger("training")

    trainer = TrainingPhase(PROCESSED_DIR, MODEL_DIR)
    trainer.run_pipeline()
