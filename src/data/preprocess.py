import os
import pandas as pd
import joblib
import numpy as np
from tools.logger import get_logger
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

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

        # Escalado / normalización
        # Separar variables objetivo y predictoras
        target_cols = ["Zone 1 Power Consumption", "Zone 2  Power Consumption", "Zone 3  Power Consumption"]
        feature_cols = [c for c in self.df.columns if c not in ["DateTime"] + target_cols]

        X = self.df[feature_cols]
        y = self.df[target_cols]

        self.scaler = StandardScaler()

        # Escalador para X
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Escalador para Y
        y_scaled = self.scaler.fit_transform(y)
        y_scaled = pd.DataFrame(y_scaled, columns=y.columns)

        # Train y temp (Train 70%, temp 30%)
        X_train, X_temp, Y_train, Y_temp = train_test_split(X_scaled, y_scaled, test_size=0.30, random_state=42)

        # Validation y Test (15% y 15%)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

        # Exportar a CSV
        X_train.to_csv(os.path.join(self.processed_dir,'X_train.csv'), index=False)
        X_val.to_csv(os.path.join(self.processed_dir,'X_val.csv'), index=False)
        X_test.to_csv(os.path.join(self.processed_dir,'X_test.csv'), index=False)

        Y_train.to_csv(os.path.join(self.processed_dir,'Y_train.csv'), index=False)
        Y_val.to_csv(os.path.join(self.processed_dir,'Y_val.csv'), index=False)
        Y_test.to_csv(os.path.join(self.processed_dir,'Y_test.csv'), index=False)
        


        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        self.df.to_csv(os.path.join(self.processed_dir, 'power_tetouan_city_ready.csv'))
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
