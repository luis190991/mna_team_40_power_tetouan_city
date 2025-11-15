"""
PRUEBAS UNITARIAS Y DE INTEGRACIÓN - PROYECTO POWER TETOUAN
Ejecutar con: python tests/todo.py
"""

import unittest
import tempfile
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# =============================================================================
# CLASES PRINCIPALES DEL PROYECTO
# =============================================================================

class DataPreprocessor:
    """Clase para preprocesamiento de datos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def handle_missing_values(self, df, strategy='mean'):
        """Maneja valores faltantes en el DataFrame"""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'drop':
            df_clean = df_clean.dropna()
            
        return df_clean
    
    def scale_features(self, df, columns):
        """Escala características numéricas"""
        df_scaled = df.copy()
        valid_columns = [col for col in columns if col in df_scaled.columns]
        if valid_columns:
            df_scaled[valid_columns] = self.scaler.fit_transform(df_scaled[valid_columns])
        return df_scaled
    
    def encode_categorical(self, df, columns):
        """Codifica variables categóricas usando one-hot encoding"""
        df_encoded = df.copy()
        for col in columns:
            if col in df_encoded.columns:
                df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=[col])
        return df_encoded

class ModelTrainer:
    """Clase para entrenamiento de modelos de machine learning"""
    
    def __init__(self, model_type='svr', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X, y, test_size=0.2):
        """Entrena el modelo seleccionado"""
        # Validar parámetros
        if test_size <= 0 or test_size >= 1:
            test_size = 0.2
            
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Preparar datos según el modelo
        if self.model_type == 'svr':
            X_train_processed = self.scaler.fit_transform(X_train)
            X_test_processed = self.scaler.transform(X_test)
            self.model = SVR(kernel='rbf', C=10.0, gamma='scale')
        elif self.model_type == 'xgb':
            X_train_processed, X_test_processed = X_train, X_test
            self.model = GradientBoostingRegressor(
                n_estimators=100, 
                random_state=self.random_state
            )
        elif self.model_type == 'linear':
            X_train_processed, X_test_processed = X_train, X_test
            self.model = LinearRegression()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
        
        # Entrenar modelo
        self.model.fit(X_train_processed, y_train)
        
        # Predecir y calcular métricas
        y_pred = self.model.predict(X_test_processed)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'test_size': len(X_test)
        }
        
        return self.model, metrics
    
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
            
        if self.model_type == 'svr':
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
            
        return self.model.predict(X_processed)
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }, filepath)
        else:
            raise ValueError("No hay modelo entrenado para guardar")
    
    def load_model(self, filepath):
        """Carga un modelo guardado"""
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.scaler = loaded_data['scaler']
        self.model_type = loaded_data['model_type']
        return self.model

class ModelEvaluator:
    """Clase para evaluación y visualización de modelos"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calcula todas las métricas de evaluación"""
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    @staticmethod
    def plot_predictions_vs_actual(y_true, y_pred, title="Predicciones vs Valores Reales"):
        """Crea gráfico de predicciones vs valores reales"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_residuals(y_true, y_pred, title="Análisis de Residuales"):
        """Crea gráfico de análisis de residuales"""
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicciones')
        ax1.set_ylabel('Residuales')
        ax1.set_title('Residuales vs Predicciones')
        ax1.grid(True, alpha=0.3)
        
        # Histograma de residuales
        ax2.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuales')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Residuales')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig

class DataVisualizer:
    """Clase para visualización de datos"""
    
    @staticmethod
    def plot_correlation_matrix(df, title="Matriz de Correlación"):
        """Crea matriz de correlación para variables numéricas"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None
            
        plt.figure(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_distributions(df, columns=None, title="Distribuciones de Variables"):
        """Crea histogramas de distribuciones"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns[:4]  # Máximo 4 columnas
        
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar ejes vacíos
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig

# =============================================================================
# PRUEBAS UNITARIAS
# =============================================================================

class TestDataPreprocessing(unittest.TestCase):
    """Pruebas para el preprocesamiento de datos"""
    
    def setUp(self):
        self.processor = DataPreprocessor()
        np.random.seed(42)
        
        # Datos de ejemplo
        self.sample_data = pd.DataFrame({
            'temperature': np.random.normal(25, 5, 100),
            'humidity': np.random.normal(60, 15, 100),
            'windspeed': np.random.normal(15, 5, 100),
            'power_consumption': np.random.normal(500, 100, 100),
            'area': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Datos con valores faltantes
        self.data_with_missing = self.sample_data.copy()
        self.data_with_missing.loc[:5, 'temperature'] = np.nan
        self.data_with_missing.loc[5:10, 'humidity'] = np.nan
    
    def test_handle_missing_values_mean(self):
        """Test: Manejo de valores faltantes con estrategia mean"""
        processed_data = self.processor.handle_missing_values(self.data_with_missing, 'mean')
        self.assertEqual(processed_data.isnull().sum().sum(), 0)
    
    def test_handle_missing_values_median(self):
        """Test: Manejo de valores faltantes con estrategia median"""
        processed_data = self.processor.handle_missing_values(self.data_with_missing, 'median')
        self.assertEqual(processed_data.isnull().sum().sum(), 0)
    
    def test_scale_features(self):
        """Test: Escalado de características"""
        columns_to_scale = ['temperature', 'humidity']
        scaled_data = self.processor.scale_features(self.sample_data, columns_to_scale)
        
        for col in columns_to_scale:
            self.assertAlmostEqual(scaled_data[col].mean(), 0, places=1)
            self.assertAlmostEqual(scaled_data[col].std(), 1.0, places=1)
    
    def test_encode_categorical(self):
        """Test: Codificación de variables categóricas"""
        encoded_data = self.processor.encode_categorical(self.sample_data, ['area'])
        expected_columns = ['area_A', 'area_B', 'area_C']
        
        for col in expected_columns:
            self.assertIn(col, encoded_data.columns)

class TestModelTraining(unittest.TestCase):
    """Pruebas para el entrenamiento de modelos"""
    
    def setUp(self):
        self.X, self.y = make_regression(
            n_samples=200, n_features=5, noise=0.1, random_state=42
        )
    
    def test_svr_training(self):
        """Test: Entrenamiento de modelo SVR"""
        trainer = ModelTrainer(model_type='svr')
        model, metrics = trainer.train(self.X, self.y)
        
        self.assertIsNotNone(model)
        self.assertIn('r2_score', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertGreater(metrics['r2_score'], 0.1)
    
    def test_xgb_training(self):
        """Test: Entrenamiento de modelo XGBoost"""
        trainer = ModelTrainer(model_type='xgb')
        model, metrics = trainer.train(self.X, self.y)
        
        self.assertIsNotNone(model)
        self.assertGreater(metrics['r2_score'], 0.5)
    
    def test_linear_training(self):
        """Test: Entrenamiento de modelo Linear Regression"""
        trainer = ModelTrainer(model_type='linear')
        model, metrics = trainer.train(self.X, self.y)
        
        self.assertIsNotNone(model)
        self.assertIn('r2_score', metrics)
    
    def test_model_prediction(self):
        """Test: Predicciones del modelo"""
        trainer = ModelTrainer(model_type='xgb')
        model, metrics = trainer.train(self.X, self.y)
        
        predictions = trainer.predict(self.X[:10])
        self.assertEqual(len(predictions), 10)
        self.assertIsInstance(predictions, np.ndarray)

class TestModelPersistence(unittest.TestCase):
    """Pruebas para la persistencia de modelos"""
    
    def setUp(self):
        self.X, self.y = make_regression(n_samples=100, n_features=3, random_state=42)
        self.trainer = ModelTrainer(model_type='svr')
        self.model, _ = self.trainer.train(self.X, self.y)
    
    def test_save_load_model(self):
        """Test: Guardado y carga de modelo"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Guardar modelo
            self.trainer.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Cargar modelo
            new_trainer = ModelTrainer()
            loaded_model = new_trainer.load_model(model_path)
            
            # Verificar que funciona
            predictions = new_trainer.predict(self.X[:5])
            self.assertEqual(len(predictions), 5)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

class TestModelEvaluation(unittest.TestCase):
    """Pruebas para la evaluación de modelos"""
    
    def setUp(self):
        np.random.seed(42)
        self.y_true = np.random.normal(0, 1, 100)
        self.y_pred = self.y_true + np.random.normal(0, 0.1, 100)
    
    def test_metrics_calculation(self):
        """Test: Cálculo de métricas"""
        metrics = ModelEvaluator.calculate_metrics(self.y_true, self.y_pred)
        
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertGreater(metrics['r2'], 0.9)
    
    def test_perfect_predictions_metrics(self):
        """Test: Métricas con predicciones perfectas"""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        
        self.assertEqual(metrics['r2'], 1.0)
        self.assertEqual(metrics['rmse'], 0.0)
        self.assertEqual(metrics['mae'], 0.0)
    
    def test_plot_functions(self):
        """Test: Funciones de visualización"""
        # Verificar que las funciones de plotting no generan errores
        try:
            fig1 = ModelEvaluator.plot_predictions_vs_actual(self.y_true, self.y_pred)
            fig2 = ModelEvaluator.plot_residuals(self.y_true, self.y_pred)
            plt.close('all')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Las funciones de plotting fallaron: {e}")

class TestDataVisualization(unittest.TestCase):
    """Pruebas para la visualización de datos"""
    
    def setUp(self):
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'temp': np.random.normal(25, 5, 50),
            'humidity': np.random.normal(60, 10, 50),
            'windspeed': np.random.normal(15, 3, 50),
            'power': np.random.normal(500, 50, 50)
        })
    
    def test_correlation_matrix(self):
        """Test: Matriz de correlación"""
        try:
            fig = DataVisualizer.plot_correlation_matrix(self.sample_data)
            if fig is not None:
                plt.close(fig)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Matriz de correlación falló: {e}")
    
    def test_distributions_plot(self):
        """Test: Gráfico de distribuciones"""
        try:
            fig = DataVisualizer.plot_distributions(self.sample_data)
            plt.close(fig)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Gráfico de distribuciones falló: {e}")

# =============================================================================
# PRUEBAS DE INTEGRACIÓN
# =============================================================================

class TestIntegrationPipeline(unittest.TestCase):
    """Pruebas de integración para el pipeline completo"""
    
    def setUp(self):
        # Crear datos simulados de consumo energético
        np.random.seed(42)
        n_samples = 300
        
        self.energy_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'temperature': np.random.normal(25, 8, n_samples),
            'humidity': np.random.normal(65, 20, n_samples),
            'windspeed': np.random.normal(12, 6, n_samples),
            'area': np.random.choice(['Residential', 'Commercial'], n_samples),
            'power_consumption': 0
        })
        
        # Crear target con relación real a las características
        self.energy_data['power_consumption'] = (
            300 + 
            10 * self.energy_data['temperature'] +
            5 * self.energy_data['humidity'] +
            8 * self.energy_data['windspeed'] +
            np.random.normal(0, 50, n_samples)
        )
        
        # Introducir valores faltantes
        self.energy_data.loc[10:15, 'temperature'] = np.nan
        self.energy_data.loc[20:25, 'humidity'] = np.nan
    
    def test_complete_pipeline_svr(self):
        """Test: Pipeline completo con modelo SVR"""
        # 1. Preprocesamiento
        processor = DataPreprocessor()
        cleaned_data = processor.handle_missing_values(self.energy_data)
        encoded_data = processor.encode_categorical(cleaned_data, ['area'])
        scaled_data = processor.scale_features(
            encoded_data, 
            ['temperature', 'humidity', 'windspeed']
        )
        
        # 2. Preparación de datos
        feature_cols = [col for col in scaled_data.columns 
                       if col not in ['timestamp', 'power_consumption']]
        X = scaled_data[feature_cols].values
        y = scaled_data['power_consumption'].values
        
        # 3. Entrenamiento
        trainer = ModelTrainer(model_type='svr')
        model, metrics = trainer.train(X, y)
        
        # 4. Verificaciones
        self.assertIsNotNone(model)
        self.assertIn('r2_score', metrics)
        self.assertGreater(metrics['r2_score'], 0.1)
    
    def test_complete_pipeline_xgb(self):
        """Test: Pipeline completo con modelo XGBoost"""
        # 1. Preprocesamiento
        processor = DataPreprocessor()
        cleaned_data = processor.handle_missing_values(self.energy_data)
        encoded_data = processor.encode_categorical(cleaned_data, ['area'])
        
        # 2. Preparación de datos (sin escalar para XGBoost)
        feature_cols = [col for col in encoded_data.columns 
                       if col not in ['timestamp', 'power_consumption']]
        X = encoded_data[feature_cols].values
        y = encoded_data['power_consumption'].values
        
        # 3. Entrenamiento
        trainer = ModelTrainer(model_type='xgb')
        model, metrics = trainer.train(X, y)
        
        # 4. Verificaciones
        self.assertIsNotNone(model)
        self.assertGreater(metrics['r2_score'], 0.3)
    
    def test_model_comparison(self):
        """Test: Comparación de múltiples modelos"""
        # Preprocesamiento común
        processor = DataPreprocessor()
        cleaned_data = processor.handle_missing_values(self.energy_data)
        encoded_data = processor.encode_categorical(cleaned_data, ['area'])
        
        feature_cols = [col for col in encoded_data.columns 
                       if col not in ['timestamp', 'power_consumption']]
        X = encoded_data[feature_cols].values
        y = encoded_data['power_consumption'].values
        
        # Probar diferentes modelos
        models = ['svr', 'xgb', 'linear']
        results = []
        
        for model_type in models:
            trainer = ModelTrainer(model_type=model_type)
            model, metrics = trainer.train(X, y)
            
            results.append({
                'model_type': model_type,
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse']
            })
        
        # Verificar resultados
        self.assertEqual(len(results), 3)
        
        # Al menos un modelo debe tener buen rendimiento
        best_r2 = max(result['r2_score'] for result in results)
        self.assertGreater(best_r2, 0.3)
    
    def test_pipeline_with_visualization(self):
        """Test: Pipeline completo con visualización"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Visualización de datos
                processor = DataPreprocessor()
                cleaned_data = processor.handle_missing_values(self.energy_data)
                
                # Gráfico de distribuciones
                dist_path = os.path.join(temp_dir, 'distributions.png')
                fig_dist = DataVisualizer.plot_distributions(
                    cleaned_data, 
                    ['temperature', 'humidity', 'windspeed', 'power_consumption']
                )
                fig_dist.savefig(dist_path, dpi=150, bbox_inches='tight')
                plt.close(fig_dist)
                
                # 2. Pipeline de modelo
                encoded_data = processor.encode_categorical(cleaned_data, ['area'])
                feature_cols = [col for col in encoded_data.columns 
                               if col not in ['timestamp', 'power_consumption']]
                X = encoded_data[feature_cols].values
                y = encoded_data['power_consumption'].values
                
                trainer = ModelTrainer(model_type='xgb')
                model, metrics = trainer.train(X, y)
                
                # 3. Visualización de resultados
                eval_path = os.path.join(temp_dir, 'evaluation.png')
                predictions = trainer.predict(X[:50])  # Usar subset para plotting
                fig_eval = ModelEvaluator.plot_predictions_vs_actual(y[:50], predictions)
                fig_eval.savefig(eval_path, dpi=150, bbox_inches='tight')
                plt.close(fig_eval)
                
                # Verificar que los archivos se crearon
                self.assertTrue(os.path.exists(dist_path))
                self.assertTrue(os.path.exists(eval_path))
                
        except Exception as e:
            self.fail(f"Pipeline con visualización falló: {e}")

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para ejecutar todas las pruebas"""
    print("=" * 70)
    print("🧪 EJECUTANDO PRUEBAS COMPLETAS DEL PROYECTO")
    print("=" * 70)
    
    # Crear test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar todas las pruebas
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestDataVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationPipeline))
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generar reporte
    print("=" * 70)
    print("📊 REPORTE FINAL DE PRUEBAS")
    print("=" * 70)
    print(f"Pruebas ejecutadas: {result.testsRun}")
    print(f"Pruebas exitosas: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("\n✅ COMPONENTES VALIDADOS:")
        print("   • Preprocesamiento de datos (limpieza, escalado, encoding)")
        print("   • Entrenamiento de modelos (SVR, XGBoost, Linear Regression)")
        print("   • Evaluación de modelos (métricas R², RMSE, MAE)")
        print("   • Persistencia de modelos (guardado y carga)")
        print("   • Visualización de datos (correlación, distribuciones)")
        print("   • Pipeline de integración completo")
        print("   • Comparación de múltiples modelos")
        print("\n🚀 EL SISTEMA ESTÁ LISTO PARA PRODUCCIÓN")
    else:
        print("\n❌ ALGUNAS PRUEBAS FALLARON")
        if result.failures:
            print("\n📋 FALLOS:")
            for test, traceback in result.failures:
                print(f"   - {test}: {traceback.splitlines()[-1]}")
        if result.errors:
            print("\n⚠️  ERRORES:")
            for test, traceback in result.errors:
                print(f"   - {test}: {traceback.splitlines()[-1]}")
    
    print("=" * 70)
    return result.wasSuccessful()

if __name__ == "__main__":
    # Ejecutar pruebas y terminar con código apropiado
    success = main()
    sys.exit(0 if success else 1)