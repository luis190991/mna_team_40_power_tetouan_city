import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Drift Visuals
class DriftUtils:
    @staticmethod
    def calculate_psi(expected, actual, buckets=10):
        """Population Stability Index (PSI)."""
        expected_perc = np.histogram(expected, bins=buckets)[0] / len(expected)
        actual_perc = np.histogram(actual, bins=buckets)[0] / len(actual)
        psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc + 1e-6))
        return psi

    @staticmethod
    def plot_feature_shift(original, drifted, feature, savepath):
        plt.figure(figsize=(8, 4))
        sns.kdeplot(original, label="Validación", fill=True)
        sns.kdeplot(drifted, label="Monitoreo (Drift)", fill=True)
        plt.title(f"Distribución Feature: {feature}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()


# carga de datos y modelos
class DataLoader:

    def __init__(self, base_path="data/processed", models_path="models"):
        self.base_path = base_path
        self.models_path = models_path

    def load_data(self):
        X_val = pd.read_csv(f"{self.base_path}/X_val.csv")
        y_val = pd.read_csv(f"{self.base_path}/Y_val.csv").squeeze()
        return X_val, y_val

    def load_models(self):
        """Carga los modelos XGB y SVR."""
        models = {}
        for root, _, files in os.walk(self.models_path):
            for f in files:
                if f.endswith(".pkl"):
                    model_name = f.replace(".pkl", "")
                    with open(os.path.join(root, f), "rb") as file:
                        models[model_name] = pickle.load(file)
        return models

# Generación del Drift
class DriftGenerator:
    def induce_feature_drift(self, X, feature):
        # Dift por univariable (cada feature)
        X_drift = X.copy()
        col = X.columns.get_loc(feature)

        # Drifts diferentes por tipo
        X_drift.iloc[:, col] = (
            # Drift escalado
            X.iloc[:, col] * np.random.uniform(0.7, 1.3)
            # Drift de la media
            + np.random.uniform(-1, 1)
        )

        # Valores nulos
        mask = np.random.rand(len(X)) < 0.10
        X_drift.loc[mask, feature] = X[feature].median()

        return X_drift

# Evalución del modelo
class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred, squared=False),
            "R2": r2_score(y_true, y_pred)
        }

# Pipeline del Drift
class DriftPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.generator = DriftGenerator()
        self.utils = DriftUtils()
        self.evaluator = ModelEvaluator()

        self.DRIFT_THRESHOLD = 0.10
        self.PSI_THRESHOLD = 0.20
        self.KS_THRESHOLD = 0.05

    def run(self):
        X_val, y_val = self.loader.load_data()
        models = self.loader.load_models()

        results = {}

        for model_name, model in models.items():
            print(f"\n==============================")
            print(f" Analizando el modelo: {model_name}")
            print(f"==============================")

            model_results = {}

            baseline_pred = model.predict(X_val)
            baseline_metrics = self.evaluator.evaluate(y_val, baseline_pred)

            for feature in X_val.columns:
                print(f"\n Generando el Drift para la feature: {feature}")

                # Crear Drift para la feature
                X_drift = self.generator.induce_feature_drift(X_val, feature)

                # Predicciones después del Drift
                pred_drift = model.predict(X_drift)
                drift_metrics = self.evaluator.evaluate(y_val, pred_drift)

                # PSI y KS
                psi = self.utils.calculate_psi(X_val[feature], X_drift[feature])
                ks_p = ks_2samp(X_val[feature], X_drift[feature]).pvalue

                # Métricas
                model_results[feature] = {
                    "Base": baseline_metrics,
                    "Drift": drift_metrics,
                    "PSI": psi,
                    "KS_pval": ks_p,
                    "Caida_Relativa": {
                        m: (baseline_metrics[m] - drift_metrics[m]) / (baseline_metrics[m] + 1e-6)
                        for m in baseline_metrics
                    },
                    "Alerta": (
                        any((baseline_metrics[m] - drift_metrics[m]) / (baseline_metrics[m] + 1e-6) > self.DRIFT_THRESHOLD
                            for m in baseline_metrics)
                        or psi > self.PSI_THRESHOLD
                        or ks_p < self.KS_THRESHOLD
                    )
                }

                # Graficos del comportamiento del drift
                os.makedirs("reports/drift_plots", exist_ok=True)
                self.utils.plot_feature_shift(
                    X_val[feature],
                    X_drift[feature],
                    feature,
                    f"reports/drift_plots/{model_name}_{feature}.png"
                )

            results[model_name] = model_results

        # Guardamos los reusltaados
        with open("reports/drift_results.json", "w") as fp:
            json.dump(results, fp, indent=4)

        print("Resultados guardados en: reports/drift_results.json")
        print("Gráficos guardados en: reports/drift_plots/")


# Ejecutar
if __name__ == "__main__":
    DriftPipeline().run()
