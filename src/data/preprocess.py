import os
import pandas as pd
import joblib
from tools.logger import get_logger
from sklearn.preprocessing import StandardScaler

class PowerPreprocessor:
    """
    Clase encargada de generar variables derivadas y preparar los datos para modelado.
    """

    def __init__(self, interim_path, processed_dir, model_dir):
        self.interim_path = interim_path
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.df = None
        self.scaler = None

    def load_data(self):
        logger.info(f"Cargando datos desde {self.interim_path}")
        self.df = pd.read_csv(self.interim_path)
        logger.info(f"Datos cargados: {self.df.shape}")

    def scale_and_split(self):
        """Escala variables numéricas y divide temporalmente el dataset."""
        logger.info("Escalando y dividiendo dataset...")
        
        train = pd.DataFrame({})
        val = pd.DataFrame({})
        test = pd.DataFrame({})
        df_final = pd.DataFrame({})

        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        train.to_csv(os.path.join(self.processed_dir, 'train.csv'))
        val.to_csv(os.path.join(self.processed_dir, 'val.csv'))
        test.to_csv(os.path.join(self.processed_dir, 'test.csv'))
        df_final.to_csv(os.path.join(self.processed_dir, 'power_tetouan_city_ready.csv'))
        logger.info("Datasets procesados y guardados")

    def run_pipeline(self):
        """Ejecuta todo el flujo de preprocesamiento."""
        self.load_data()
        self.scale_and_split()
        logger.info("Preprocesamiento completado")


if __name__ == "__main__":
    INTERIM_PATH = "data/interim/power_tetouan_city_clean.csv"
    PROCESSED_DIR = "data/processed"
    MODEL_DIR = "models"

    logger = get_logger("preProcessor")

    preprocessor = PowerPreprocessor(INTERIM_PATH, PROCESSED_DIR, MODEL_DIR)
    preprocessor.run_pipeline()
