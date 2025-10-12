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
        # Se eliminan los nulos.
        self.df = self.df.dropna()
        # Se eliminan los duplicados
        self.df = self.df.drop_duplicates()

        # Convertimos DateTime a formato datetime real
        self.df["DateTime"] = pd.to_datetime(self.df["DateTime"], errors="coerce")

        # Creamos solo las columnas que nos interesan
        self.df["hour"] = self.df["DateTime"].dt.hour
        self.df["day"] = self.df["DateTime"].dt.day
        self.df["month"] = self.df["DateTime"].dt.month
        self.df["year"] = self.df["DateTime"].dt.year

        # Eliminamos las columnas no necesarias
        self.df = self.df.drop(columns=["DateTime", "mixed_type_col"], errors="ignore")

        # 🔹 Convertimos las demás columnas a numéricas (por si quedaron como texto)
        for col in self.df.columns:
            self.df[col] = (
                self.df[col].astype(str).str.replace(",", ".")
            )
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        logger.info("Limpieza básica completada")

    def remove_outliers(self):
        """Detecta y corrige valores atípicos simples."""
        Outliers = self.eliminarOutliers()
        logger.info(f"Se han identificado {Outliers.shape[0]} valores atípicos.")

        # Eliminar Outliers / Otra vez valores nulos
        self.df = self.df.drop(Outliers.index).dropna()
        
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
    
    def eliminarOutliers(self):
        df = self.df
        outliers = pd.DataFrame(columns=df.columns)
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR

            outliers_col = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
            outliers = pd.concat([outliers, outliers_col])
        return outliers.drop_duplicates()


if __name__ == "__main__":
    RAW_PATH = "data/raw/power_tetouan_city_modified.csv"
    INTERIM_PATH = "data/interim/power_tetouan_city_clean.csv"

    logger = get_logger("cleaner")
    cleaner = Cleaner(RAW_PATH, INTERIM_PATH)
    cleaner.run_pipeline()
