# 📊 Proyecto de MLops (Power Tetouan City)
** Instituto Tecnológico y de Estudios Superiores de Monterrey (ITESM) **
** Maestría en Inteligencia Artificial Aplicada **     
** Curso:** Operaciones de Aprendizaje Automático ** 

---

## 🧠 Descripción del Proyecto

Este proyecto tiene como objetivo demostrar la aplicación de **Data Version Control (DVC)** dentro de un flujo de trabajo de *Machine Learning* siguiendo las mejores prácticas de ingeniería reproducible.

A través de la integración con **Git**, se asegura el versionamiento tanto del código como de los datos y modelos, permitiendo una trazabilidad completa de los experimentos y facilitando la colaboración entre miembros del equipo.

El proyecto incluye las fases de:
1. **Ingesta y limpieza de datos.**
2. **Preprocesamiento y transformación.**
3. **Entrenamiento y evaluación de modelos.**
4. **Almacenamiento y versionamiento en DVC.**

---

## ⚙️ Estructura del Proyecto

```
    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── models  <- compiled model .pkl or HDFS or .pb format
    ├── config  <- any configuration files
    ├── data
    │   ├── interim <- data in intermediate processing stage
    │   ├── processed <- data after all preprocessing has been done
    │   └── raw <- original unmodified data acting as source of truth and provenance
    ├── docs  <- usage documentation or reference papers
    ├── notebooks <- jupyter notebooks for exploratory analysis and explanation 
    ├── reports <- generated project artefacts eg. visualisations or tables
    │   └── figures
    └── src
        ├── data-proc <- scripts for processing data eg. transformations, dataset merges etc. 
        ├── viz  <- scripts for visualisation during EDA, modelling, error analysis etc. 
        ├── modeling    <- scripts for generating models
    |--- environment.yml <- file with libraries and library versions for recreating the analysis environment
    |--- dvc.yaml <- file with process specification.
```

---

## 🚀 Funcionamiento del Proyecto

### 1. Inicializar DVC
```bash
dvc init
git add .dvc .gitignore
git commit -m "Inicializa control de versiones con DVC"
```

### 2. Agregar datos al seguimiento de DVC
```bash
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Agrega datos crudos al seguimiento de DVC"
```

### 3. Definir pipeline de procesamiento y modelado
En el archivo `dvc.yaml` se declaran las etapas: clean, preprocess y train


### 4. Ejecutar pipeline completo
```bash
dvc repro
```

### 5. Sincronizar los datos con almacenamiento remoto (AWS S3)
```bash
dvc remote add -d myremote s3://nombre-del-bucket
dvc push
```

### 6. Ver versiones de datos y modelos
```bash
dvc metrics show
dvc diff
```

---

## 🧩 Tecnologías Utilizadas

- **Python 3.10+**
- **DVC**
- **Git / GitHub**
- **AWS S3 (para almacenamiento remoto)**
- **Pandas, NumPy, Scikit-learn**
- **Cookiecutter ML Project Template**

---

## 👥 Autores

| Nombre | Matrícula | Rol |
|-----------------------------|-------------|-------------|
| Franco Quintanilla Fuentes | A00826953 | Data Scientist |
| Daniel Nuñez Constantino | A01379717 | Data Engineer |
| Luis Antonio Ramírez Martínez | A01796272 | Site Reliability Engineering |
| Paulina Paz Hernández | A01652337 | Software Engineer |
| Gabriel Leal Cantú | A01282101 | ML Engineer |

Equipo #40

---

## 📚 Licencia

Este proyecto es de uso académico y se distribuye bajo la licencia MIT.  
Puedes modificar y reutilizar el código citando a los autores originales.

---

## 📦 Ejemplo de Ejecución

```bash
# Clonar el repositorio
git clone https://github.com/usuario/proyecto-dvc.git
cd proyecto-dvc

# Instalar dependencias
conda env create -f environment.yml
conda activate proyecto-dvc

# Ejecutar pipeline
dvc repro

# Subir datos y modelos al remoto
dvc push
```
