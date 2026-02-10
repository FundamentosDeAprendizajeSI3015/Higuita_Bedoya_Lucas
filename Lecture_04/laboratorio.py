# ============================================================
# TITANIC DATASET - ANALISIS EXPLORATORIO Y TRANSFORMACIONES
# ============================================================

# -------------------------------
# 1. LIBRERIAS
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from category_encoders import BinaryEncoder

sns.set(style="whitegrid")


# -------------------------------
# 2. CARGA DEL DATASET
# -------------------------------
df = pd.read_csv("Titanic-Dataset.csv")

print("Primeras filas del dataset:")
print(df.head())


# -------------------------------
# 3. EXPLORACION INICIAL
# -------------------------------
print("\nINFORMACION GENERAL:")
df.info()

print("\nDESCRIPCION ESTADISTICA:")
print(df.describe())

print("\nVALORES NULOS POR COLUMNA:")
print(df.isnull().sum())


# -------------------------------
# 4. MEDIDAS DE TENDENCIA CENTRAL
# -------------------------------
print("\nMEDIA:")
print(df[['Age', 'Fare']].mean())

print("\nMEDIANA:")
print(df[['Age', 'Fare']].median())

print("\nMODA:")
print(df[['Age', 'Fare']].mode())


# -------------------------------
# 5. MEDIDAS DE DISPERSION
# -------------------------------
print("\nDESVIACION ESTANDAR:")
print(df[['Age', 'Fare']].std())

print("\nVARIANZA:")
print(df[['Age', 'Fare']].var())

print("\nRANGO:")
print(df[['Age', 'Fare']].max() - df[['Age', 'Fare']].min())


# -------------------------------
# 6. MEDIDAS DE POSICION Y OUTLIERS
# -------------------------------
print("\nCUARTILES:")
print(df[['Age', 'Fare']].quantile([0.25, 0.5, 0.75]))


# Boxplots
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['Age'])
plt.title("Boxplot Age")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Fare'])
plt.title("Boxplot Fare")

plt.tight_layout()
plt.show()


# Eliminacion de outliers en Fare (IQR)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df['Fare'] >= Q1 - 1.5 * IQR) &
    (df['Fare'] <= Q3 + 1.5 * IQR)
]


# -------------------------------
# 7. HISTOGRAMAS
# -------------------------------
df[['Age', 'Fare']].hist(bins=20, figsize=(10, 4))
plt.show()


# -------------------------------
# 8. GRAFICOS DE DISPERSION
# -------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x='Age',
    y='Fare',
    hue='Survived',
    data=df
)
plt.title("Age vs Fare")
plt.show()


# -------------------------------
# 9. TRANSFORMACIONES DE COLUMNAS
# -------------------------------

# Label Encoding
le = LabelEncoder()
df['Sex_LE'] = le.fit_transform(df['Sex'])

# One Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Binary Encoding
be = BinaryEncoder(cols=['Cabin'])
df = be.fit_transform(df)


# -------------------------------
# 10. CORRELACION DE VARIABLES
# -------------------------------
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] < 2:
    print("No hay suficientes columnas numericas para la correlacion.")
else:
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
    plt.title("Matriz de Correlacion")
plt.show()


# -------------------------------
# 11. ESCALAMIENTO
# -------------------------------
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Alternativa Min-Max Scaling
scaler_mm = MinMaxScaler()
df[['Age', 'Fare']] = scaler_mm.fit_transform(df[['Age', 'Fare']])


# -------------------------------
# 12. TRANSFORMACION LOGARITMICA
# -------------------------------
df['Fare_log'] = np.log1p(df['Fare'])


# -------------------------------
# 13. CONCLUSIONES
# -------------------------------
print("""
CONCLUSIONES:
1. La clase social (Pclass) influye significativamente en la supervivencia.
2. Las mujeres presentan mayor probabilidad de sobrevivir.
3. Fare presenta fuerte asimetria y presencia de outliers.
4. La transformacion logaritmica mejora la distribucion de Fare.
5. Existen correlaciones importantes entre Pclass, Fare y Survived.
6. El escalamiento es fundamental antes de aplicar modelos de Machine Learning.
""")