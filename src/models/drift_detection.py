import os
import json
import pickle
import joblib
import traceback
import re
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

class DriftUtils:
    @staticmethod
    def calculate_psi(expected, actual, buckets=10):
        exp = np.asarray(expected).ravel()
        act = np.asarray(actual).ravel()
        if len(exp) == 0 or len(act) == 0:
            return 0.0
        if np.nanstd(exp) == 0 or np.nanstd(act) == 0:
            return 0.0
        eps = 1e-6
        bins = np.linspace(min(np.nanmin(exp), np.nanmin(act)),
                           max(np.nanmax(exp), np.nanmax(act)) + eps,
                           buckets + 1)
        expected_perc = np.histogram(exp, bins=bins)[0] / (len(exp) + eps)
        actual_perc = np.histogram(act, bins=bins)[0] / (len(act) + eps)
        expected_perc = np.clip(expected_perc, eps, None)
        actual_perc = np.clip(actual_perc, eps, None)
        psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
        return float(psi)

    @staticmethod
    def plot_feature_shift(original, drifted, feature, savepath):
        try:
            plt.figure(figsize=(8, 4))
            orig = pd.Series(original).dropna().astype(float)
            drf = pd.Series(drifted).dropna().astype(float)
            if orig.nunique() < 3 or drf.nunique() < 3:
                plt.hist(orig, bins=30, alpha=0.6, label="Validación", density=True)
                plt.hist(drf, bins=30, alpha=0.4, label="Monitoreo (Drift)", density=True)
            else:
                sns.kdeplot(orig, label="Validación", fill=True)
                sns.kdeplot(drf, label="Monitoreo (Drift)", fill=True)
            plt.title(f"Distribución Feature: {feature}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(savepath)
            plt.close()
        except Exception:
            print(f"Warning: fallo al generar plot para {feature}")
            traceback.print_exc()


class DataLoader:
    def __init__(self, base_path="data/processed", models_path="models"):
        self.base_path = base_path
        self.models_path = models_path

    def load_data(self):
        X_val_path = os.path.join(self.base_path, "X_val.csv")
        Y_val_path = os.path.join(self.base_path, "Y_val.csv")

        if not os.path.exists(X_val_path) or not os.path.exists(Y_val_path):
            raise FileNotFoundError(f"No se encontraron X_val o Y_val en {self.base_path}")

        X_val = pd.read_csv(X_val_path)
        Y_val = pd.read_csv(Y_val_path)  # keep as DataFrame with 3 columns
        return X_val, Y_val

    def load_models(self):
        models = {}
        for root, _, files in os.walk(self.models_path):
            for f in files:
                if not f.endswith(".pkl"):
                    continue
                path = os.path.join(root, f)
                # scaler (joblib)
                if f == "scaler.pkl":
                    try:
                        models["scaler"] = joblib.load(path)
                        print(f"Scaler cargado: {path}")
                    except Exception:
                        try:
                            with open(path, "rb") as ff:
                                models["scaler"] = pickle.load(ff)
                            print(f"Scaler cargado con pickle: {path}")
                        except Exception:
                            print(f"Warning: no se pudo cargar scaler: {path}")
                            traceback.print_exc()
                    continue

                # try joblib then pickle
                try:
                    loaded = joblib.load(path)
                    loader_used = "joblib"
                except Exception:
                    try:
                        with open(path, "rb") as ff:
                            loaded = pickle.load(ff)
                        loader_used = "pickle"
                    except Exception as e:
                        print(f"Error cargando {path}: {e}")
                        traceback.print_exc()
                        continue

                if hasattr(loaded, "predict"):
                    model_name = os.path.splitext(f)[0]
                    models[model_name] = loaded
                    print(f"Cargado modelo ({loader_used}): {model_name} <- {path}")
                else:
                    print(f"Ignorado (no es modelo con predict): {path}")
        return models


class DriftGenerator:
    def induce_feature_drift(self, X, feature):
        if feature not in X.columns:
            raise KeyError(feature)
        X_drift = X.copy()
        if not pd.api.types.is_numeric_dtype(X_drift[feature]):
            try:
                X_drift[feature] = pd.to_numeric(X_drift[feature], errors="coerce")
            except Exception:
                return X_drift
        col = X[feature].astype(float)
        scale = np.random.uniform(0.7, 1.3)
        shift = np.random.uniform(-1, 1)
        new_vals = col * scale + shift
        mask = np.random.rand(len(X)) < 0.10
        new_vals.loc[mask] = col.median()
        X_drift.loc[:, feature] = new_vals
        return X_drift


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Compatible con versiones antiguas de sklearn: calcular RMSE como sqrt(MSE).
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)  # compatible con todas las versiones
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


class DriftPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.generator = DriftGenerator()
        self.utils = DriftUtils()
        self.evaluator = ModelEvaluator()
        self.DRIFT_THRESHOLD = 0.10
        self.PSI_THRESHOLD = 0.20
        self.KS_THRESHOLD = 0.05

        # Targets mapping based on lo que me indicaste
        self.zone_target_map = {
            1: "Zone 1 Power Consumption",
            2: "Zone 2  Power Consumption",  # nota: tu CSV tiene doble espacio en "Zone 2  Power Consumption"
            3: "Zone 3  Power Consumption"
        }

        # features esperadas (según lo que me indicaste)
        self.feature_cols = [
            "Temperature", "Humidity", "Wind Speed", "general diffuse flows",
            "diffuse flows", "hour", "day", "month", "year"
        ]

    def _to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.ndarray, list, tuple, pd.Series)):
            return [self._to_json_serializable(x) for x in obj]
        return obj

    def _extract_zone_from_name(self, model_name):
        m = re.search(r'zone[_]*([1-3])', model_name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def run(self):
        X_val, Y_val = self.loader.load_data()

        # Ensure expected feature columns exist in X_val
        missing_features = [c for c in self.feature_cols if c not in X_val.columns]
        if missing_features:
            print(f"Warning: faltan features esperadas en X_val: {missing_features}")
            # No abortamos; vamos a usar las columnas disponibles
            available_features = [c for c in self.feature_cols if c in X_val.columns]
        else:
            available_features = self.feature_cols.copy()

        # Build a single dataframe that preserves alignment between X and Y
        df_val = pd.concat([X_val.reset_index(drop=True), Y_val.reset_index(drop=True)], axis=1)

        models = self.loader.load_models()
        results = {}

        for model_name, model in models.items():
            if model_name == "scaler":
                continue

            print(f"\n==============================")
            print(f" Analizando el modelo: {model_name}")
            print(f"==============================")

            zone = self._extract_zone_from_name(model_name)
            if zone is None:
                print(f"⚠️ No se pudo extraer zona desde el nombre {model_name}. Se omite.")
                continue

            target_col = self.zone_target_map.get(zone)
            if target_col not in df_val.columns:
                print(f"⚠️ Target {target_col} no se encuentra en Y_val. Columnas disponibles: {list(Y_val.columns)}. Se omite {model_name}.")
                continue

            # Prepare X and y for this zone preserving exact row alignment
            X_zone = df_val[available_features].copy()
            y_zone = df_val[target_col].copy()

            # If model exposes feature_names_in_, try to reorder/select columns accordingly
            try:
                if hasattr(model, "feature_names_in_"):
                    expected_features = list(model.feature_names_in_)
                    # select intersection in the order expected by the model
                    intersect = [c for c in expected_features if c in X_zone.columns]
                    if len(intersect) == 0:
                        print(f"Warning: none of model's feature_names_in_ found in X_val for {model_name}. Using available columns.")
                    else:
                        X_zone = X_zone[intersect]
                else:
                    # if model doesn't expose, attempt to use exactly available_features (intersection)
                    X_zone = X_zone[[c for c in available_features if c in X_zone.columns]]
            except Exception:
                traceback.print_exc()

            # final shape check: if X_zone empty -> skip
            if X_zone.shape[0] == 0:
                print(f"⚠️ X_zone está vacío para {model_name}. Se omite.")
                continue

            # baseline prediction
            baseline_pred = None
            try:
                baseline_pred = model.predict(X_zone)
            except Exception as e:
                print(f"Error al predecir baseline con {model_name}: {e}")
                traceback.print_exc()
                # try fallback: if the model expects numpy array with shape (n_samples, n_features)
                try:
                    baseline_pred = model.predict(X_zone.values)
                except Exception:
                    print(f"Fatal: no se pudo predecir con {model_name}. Se omite.")
                    continue

            # Normalize shapes
            baseline_pred = np.asarray(baseline_pred).ravel()
            y_zone_arr = np.asarray(y_zone).ravel()

            # Debug print de shapes (útiles si algo sale mal)
            print("SHAPES DEBUG:")
            print(" len(X_zone):", X_zone.shape)
            print(" len(y_zone):", y_zone_arr.shape)
            print(" len(baseline_pred):", baseline_pred.shape)

            if baseline_pred.shape[0] != y_zone_arr.shape[0]:
                print(f"ERROR: mismatch de longitudes para {model_name}: y ({y_zone_arr.shape[0]}) vs preds ({baseline_pred.shape[0]}).")
                continue

            # calcula métricas baseline
            baseline_metrics = self.evaluator.evaluate(y_zone_arr, baseline_pred)

            # iterar features y generar drift por cada feature numérica
            model_results = {}
            for feature in X_zone.columns:
                print(f"\n Generando Drift para feature: {feature}")
                try:
                    X_drift = self.generator.induce_feature_drift(X_zone, feature)
                except Exception:
                    print(f"Warning: fallo induciendo drift en {feature}, usando copia sin cambios.")
                    traceback.print_exc()
                    X_drift = X_zone.copy()

                # predecir con X_drift
                try:
                    pred_drift = model.predict(X_drift)
                except Exception:
                    try:
                        pred_drift = model.predict(X_drift.values)
                    except Exception:
                        print(f"Warning: no se pudo predecir con X_drift para {model_name} - {feature}. Se saltará esta feature.")
                        traceback.print_exc()
                        continue

                pred_drift = np.asarray(pred_drift).ravel()
                if pred_drift.shape[0] != y_zone_arr.shape[0]:
                    print(f"Warning: pred_drift tiene distinto tamaño ({pred_drift.shape[0]}) que y ({y_zone_arr.shape[0]}). Skipping feature {feature}.")
                    continue

                drift_metrics = self.evaluator.evaluate(y_zone_arr, pred_drift)

                # PSI y KS
                psi = None
                ks_p = None
                try:
                    if pd.api.types.is_numeric_dtype(X_zone[feature]):
                        psi = self.utils.calculate_psi(X_zone[feature].fillna(0).astype(float), X_drift[feature].fillna(0).astype(float))
                        ks_p = float(ks_2samp(X_zone[feature].dropna().astype(float), X_drift[feature].dropna().astype(float)).pvalue)
                    else:
                        psi = 0.0
                        ks_p = 1.0
                except Exception:
                    traceback.print_exc()
                    psi = None
                    ks_p = None

                # caida relativa
                caida_rel = {}
                for m in baseline_metrics:
                    bv = baseline_metrics.get(m)
                    dv = drift_metrics.get(m) if drift_metrics.get(m) is not None else bv
                    try:
                        caida_rel[m] = (bv - dv) / (abs(bv) + 1e-6)
                    except Exception:
                        caida_rel[m] = None

                metric_drop_flag = any(
                    (baseline_metrics[m] is not None and drift_metrics.get(m) is not None and
                     (baseline_metrics[m] - drift_metrics[m]) / (abs(baseline_metrics[m]) + 1e-6) > self.DRIFT_THRESHOLD)
                    for m in baseline_metrics
                )
                psi_flag = (psi is not None and psi > self.PSI_THRESHOLD)
                ks_flag = (ks_p is not None and ks_p < self.KS_THRESHOLD)
                alerta = metric_drop_flag or psi_flag or ks_flag

                # plots
                os.makedirs("reports/drift_plots", exist_ok=True)
                plot_path = f"reports/drift_plots/{model_name}_{feature}.png"
                try:
                    self.utils.plot_feature_shift(X_zone[feature], X_drift[feature], feature, plot_path)
                except Exception:
                    traceback.print_exc()

                model_results[feature] = {
                    "Base": baseline_metrics,
                    "Drift": drift_metrics,
                    "PSI": psi,
                    "KS_pval": ks_p,
                    "Caida_Relativa": caida_rel,
                    "Alerta": alerta,
                    "Plot": plot_path
                }

            results[model_name] = model_results

        # Guardar resultados serializables
        serializable = self._to_json_serializable(results)
        os.makedirs("reports", exist_ok=True)
        with open("reports/drift_results.json", "w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=4, ensure_ascii=False)

        print("Resultados guardados en: reports/drift_results.json")
        print("Gráficos guardados en: reports/drift_plots/")


if __name__ == "__main__":
    DriftPipeline().run()
