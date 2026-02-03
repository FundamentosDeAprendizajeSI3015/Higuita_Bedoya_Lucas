
# =============================================================
# LABORATORIO DE PREPROCESAMIENTO Y EXPLORACIÓN INICIAL
# Dataset fuente: IRIS (scikit-learn)
# =============================================================
# Este archivo contiene ejemplos estructurados para:
# 1) Carga del dataset
# 2) Exploración inicial (EDA básico)
# 3) Limpieza mínima
# 4) Codificación y escalado
# 5) Partición Train/Test
# 6) Exportación del dataset limpio
# =============================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# -------------------------------------------------------------
# 1. CARGA DEL DATASET
# -------------------------------------------------------------
print("Cargando dataset Iris...")
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Primeras filas:")
print(df.head())

# -------------------------------------------------------------
# 2. EXPLORACIÓN INICIAL
# -------------------------------------------------------------
print("Información del dataset:")
print(df.info())

print("Descripción estadística:")
print(df.describe())

print("Valores únicos por columna:")
print(df.nunique())

# -------------------------------------------------------------
# 3. LIMPIEZA MÍNIMA
# -------------------------------------------------------------
print("Verificando valores nulos:")
print(df.isna().sum())

# En Iris no hay nulos, pero ejemplificamos imputación:
if df.isna().sum().sum() > 0:
    df = df.fillna(df.mean(numeric_only=True))

# Outliers: detección simple con z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outlier_mask = (z_scores > 3).any(axis=1)
print(f"Registros detectados como posibles outliers: {outlier_mask.sum()}")

# No se eliminarán por defecto, pero podría hacerse:
# df = df[~outlier_mask]

# -------------------------------------------------------------
# 4. CODIFICACIÓN Y ESCALADO
# -------------------------------------------------------------
X = df.drop(columns=['target'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Ejemplo de datos escalados:")
print(X_scaled.head())

# Convierte todos los numeros a una escala entre 0 y 1. 

# -------------------------------------------------------------
# 5. PARTICIÓN TRAIN/TEST
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Tamaños de los conjuntos:")
print("Train:", X_train.shape, " Test:", X_test.shape)

# Divide los datos en conjuntos de entrenamiento y prueba, manteniendo la proporción de clases

# -------------------------------------------------------------
# 6. EXPORTACIÓN
# -------------------------------------------------------------
output_dir = Path("./data_output")
output_dir.mkdir(exist_ok=True)

train_path = output_dir / "iris_train.parquet"
test_path = output_dir / "iris_test.parquet"

X_train.assign(target=y_train).to_parquet(train_path, index=False)
X_test.assign(target=y_test).to_parquet(test_path, index=False)

print("Archivos exportados:")
print(train_path)
print(test_path)

print("Laboratorio finalizado correctamente.")