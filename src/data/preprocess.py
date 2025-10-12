import os
import pandas as pd
import joblib
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
        print(f"Cargando datos desde {self.interim_path}")
        self.df = pd.read_csv(self.interim_path)
        print(f"Datos cargados: {self.df.shape}")

    # ----------------------------------------------------------
    def feature_engineering(self):
        """Genera variables de tiempo, índices y rezagos."""
        print("Generando nuevas características...")

        print(f"Nuevas columnas generadas: {self.df.shape[1]} totales")

    # ----------------------------------------------------------
    def scale_and_split(self):
        """Escala variables numéricas y divide temporalmente el dataset."""
        print("Escalando y dividiendo dataset...")
        
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
        print("Datasets procesados y guardados")

    # ----------------------------------------------------------
    def run_pipeline(self):
        """Ejecuta todo el flujo de preprocesamiento."""
        self.load_data()
        self.feature_engineering()
        self.scale_and_split()
        print("Preprocesamiento completado")


if __name__ == "__main__":
    INTERIM_PATH = "data/interim/power_tetouan_city_clean.csv"
    PROCESSED_DIR = "data/processed"
    MODEL_DIR = "models"

    preprocessor = PowerPreprocessor(INTERIM_PATH, PROCESSED_DIR, MODEL_DIR)
    preprocessor.run_pipeline()
