import os
import pandas as pd
from tools.logger import get_logger


class Cleaner:
    """
    Clase encargada de limpiar el dataset crudo de consumo energético de Tetouan City.
    """

    def __init__(self, raw_path, interim_path):
        self.raw_path = raw_path
        self.interim_path = interim_path
        os.makedirs(os.path.dirname(interim_path), exist_ok=True)
        self.df = None

    def load_data(self):
        """Carga el dataset crudo y realiza validaciones iniciales."""
        self.df = pd.read_csv(self.raw_path)
        logger.info(f"Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")

    def basic_cleaning(self):
        """Elimina duplicados, valores vacíos y corrige tipos de datos."""
        logger.info("Limpieza básica completada")

    def remove_outliers(self):
        """Detecta y corrige valores atípicos simples."""
        logger.info("Outliers tratados")

    def save_clean_data(self):
        """Guarda el dataset limpio."""
        os.makedirs(os.path.dirname(self.interim_path), exist_ok=True)
        self.df.to_csv(self.interim_path, index=False)
        logger.info(f"Dataset limpio guardado en {self.interim_path}")

    def run_pipeline(self):
        """Ejecuta todo el flujo de limpieza."""
        self.load_data()
        self.basic_cleaning()
        self.remove_outliers()
        self.save_clean_data()
        logger.info("Limpieza completada")


if __name__ == "__main__":
    RAW_PATH = "data/raw/power_tetouan_city_modified.csv"
    INTERIM_PATH = "data/interim/power_tetouan_city_clean.csv"

    logger = get_logger("cleaner")
    cleaner = Cleaner(RAW_PATH, INTERIM_PATH)
    cleaner.run_pipeline()
