import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuración estética de gráficos
plt.style.use("ggplot")

# Cargar CSV
df = pd.read_csv("fintech_top_sintetico_2025.csv")

# ==========================================================
# INSPECCIÓN RÁPIDA DEL DATASET
# ==========================================================

print("Primeras filas:")
print(df.head())

print("\nÚltimas filas:")
print(df.tail())

print("\nInformación general:")
print(df.info())

print("\nDescripción estadística:")
print(df.describe(include="all"))

print("\nDimensiones del dataset:")
print(df.shape)

# ==========================================================
# MANEJO DE VALORES FALTANTES (NaNs)
# ==========================================================

print("\nValores faltantes por columna:")
print(df.isna().sum())

# Rellenar valores numéricos con la mediana de cada columna
for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].median())

# Rellenar valores categóricos con el valor más frecuente (moda)
for col in df.select_dtypes(include="object"):
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nVerificación después de imputación:")
print(df.isna().sum())

# ==========================================================
# LIMPIEZA DE TEXTO Y DUPLICADOS
# ==========================================================

# Convertir texto a minúsculas y eliminar espacios en blanco al inicio/final
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip().str.lower()

# Eliminar filas completamente duplicadas
df.drop_duplicates(inplace=True)

# ==========================================================
# VALIDACIÓN LÓGICA Y CONSISTENCIA
# ==========================================================

# Eliminar filas con valores negativos en columnas numéricas
for col in df.select_dtypes(include=np.number):
    df = df[df[col] >= 0]

# Eliminar columnas no numéricas que no aportan valor al modelo
df = df.drop(columns=["Month", "Company", "Ticker"], errors="ignore")

# ==========================================================
# AGREGACIÓN POR REGIÓN Y SEGMENTO
# ==========================================================

# Agrupar datos por región y calcular ingresos promedio
if "Region" in df.columns and "Revenue_USD_M" in df.columns:
    grouped = df.groupby("Region")["Revenue_USD_M"].mean()
    print("\nIngreso promedio por región:")
    print(grouped)

# Agrupar datos por segmento y calcular TPV promedio
if "Segment" in df.columns and "TPV_USD_B" in df.columns:
    grouped = df.groupby("Segment")["TPV_USD_B"].mean()
    print("\nTPV promedio por segmento:")
    print(grouped)

# ==========================================================
# ONE HOT ENCODING (VARIABLES CATEGÓRICAS REALES)
# ==========================================================

# Codificar variables categóricas útiles: Country, Region, Segment, Subsegment
columnas_categoricas_utiles = [col for col in ["Country", "Region", "Segment", "Subsegment"] if col in df.columns]

# Convertir variables categóricas en variables binarias (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=columnas_categoricas_utiles, drop_first=True)

print("\nColumnas después de One Hot Encoding:")
print(list(df_encoded.columns)[:20], "...")  # Mostrar solo las primeras 20 columnas

# ==========================================================
# MEDIDAS DE TENDENCIA CENTRAL
# ==========================================================

print("\nMedia:")
print(df.mean(numeric_only=True))

print("\nMediana:")
print(df.median(numeric_only=True))

print("\nModa:")
print(df.mode().iloc[0])

# ==========================================================
# MEDIDAS DE DISPERSIÓN
# ==========================================================

print("\nVarianza:")
print(df.var(numeric_only=True))

print("\nDesviación estándar:")
print(df.std(numeric_only=True))

print("\nRango:")
numeric_df = df.select_dtypes(include=np.number)
print(numeric_df.max() - numeric_df.min())

# ==========================================================
# DETECCIÓN Y ELIMINACIÓN DE OUTLIERS (IQR)
# ==========================================================

def remove_outliers_iqr(dataframe):
    # Función para detectar y eliminar outliers usando el método de Rango Intercuartil (IQR)
    df_out = dataframe.copy()
    for col in df_out.select_dtypes(include=np.number):
        # Calcular cuartiles Q1 (25%) y Q3 (75%)
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calcular límites: outliers son valores fuera de [Q1-1.5*IQR, Q3+1.5*IQR]
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Filtrar para mantener solo valores dentro de los límites
        df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
    
    return df_out

df_no_outliers = remove_outliers_iqr(df)

print("\nDimensiones después de eliminar outliers:")
print(df_no_outliers.shape)

# ==========================================================
# HISTOGRAMAS
# ==========================================================

# Crear histogramas para todas las columnas numéricas
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# ==========================================================
# GRÁFICOS DE DISPERSIÓN
# ==========================================================

# Obtener las dos primeras columnas numéricas para graficar su relación
numeric_cols = df.select_dtypes(include=np.number).columns

if len(numeric_cols) >= 2:
    # Crear gráfico de dispersión entre las dos primeras variables numéricas
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title("Gráfico de Dispersión")
    plt.show()

# ==========================================================
# BOX PLOTS POR SEGMENTO
# ==========================================================

# Visualizar distribución de Revenue por Segment para detectar outliers y patrones
if "Segment" in df.columns and "Revenue_USD_M" in df.columns:
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="Segment", y="Revenue_USD_M", palette="Set2")
    plt.title("Distribución de Ingresos por Segmento")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ingresos (USD Millones)")
    plt.tight_layout()
    plt.show()

# ==========================================================
# INGRESOS PROMEDIO POR REGIÓN (GRÁFICO DE BARRAS)
# ==========================================================

# Calcular ingresos promedio por región y visualizar en gráfico de barras
if "Region" in df.columns and "Revenue_USD_M" in df.columns:
    revenue_by_region = df.groupby("Region")["Revenue_USD_M"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    revenue_by_region.plot(kind="bar", color="steelblue")
    plt.title("Ingresos Promedio por Región")
    plt.xlabel("Región")
    plt.ylabel("Ingresos Promedio (USD Millones)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ==========================================================
# LABEL ENCODING
# ==========================================================

# Crear copia del dataset para aplicar codificación de etiquetas
df_label = df.copy()
le = LabelEncoder()

# Convertir valores categóricos en números (0, 1, 2, etc.)
for col in df_label.select_dtypes(include="object"):
    df_label[col] = le.fit_transform(df_label[col])

# ==========================================================
# CORRELACIÓN (CON VALORES ANOTADOS)
# ==========================================================

# Calcular y visualizar matriz de correlación entre todas las variables con valores numéricos
plt.figure(figsize=(12, 10))
sns.heatmap(df_label.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Correlación"})
plt.title("Matriz de Correlación (Valores Anotados)")
plt.tight_layout()
plt.show()

# ==========================================================
# ESCALAMIENTO
# ==========================================================

# Normalizar todas las variables numéricas a media 0 y desviación estándar 1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_label)

print("\nEscalamiento aplicado con StandardScaler.")

# ==========================================================
# TRANSFORMACIÓN LOGARÍTMICA
# ==========================================================

# Crear copia del dataset para aplicar transformación logarítmica
df_log = df.copy()

# Aplicar logaritmo natural a columnas positivas (reduce distribuciones asimétricas)
for col in df_log.select_dtypes(include=np.number):
    if (df_log[col] > 0).all():
        df_log[col] = np.log1p(df_log[col])