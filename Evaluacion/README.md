# Evaluacion - Pipeline de Analisis Fintech 2025

## Descripcion del proyecto
Este proyecto realiza el preprocesamiento y analisis exploratorio del dataset
fintech_top_sintetico_2025.csv. El objetivo es dejar los datos listos para
modelado, aplicando limpieza, imputacion de faltantes, validaciones y
transformaciones, ademas de generar graficas clave para entender el
comportamiento de las variables.

## Dataset
Archivo: fintech_top_sintetico_2025.csv
Contenido: métricas mensuales de empresas fintech en 2025, incluyendo usuarios,
TPV, revenue, ARPU, churn, CAC, y variables categoricas como Region, Segment y
Subsegment.

## Como ejecutar
1. Entrar a la carpeta Evaluacion.
2. Ejecutar el script:
   python pipeline.py
3. Las graficas se guardan en la carpeta output.

## Que hace el pipeline
- Inspeccion inicial del dataset (head, tail, info, describe, shape)
- Imputacion de faltantes:
  - Numericos con la mediana
  - Categoricos con la moda
- Limpieza de texto (minusculas y espacios) y eliminacion de duplicados
- Validaciones: elimina valores negativos en columnas numericas
- Eliminacion de columnas no informativas para el modelo
- Agregaciones por Region y Segment para analisis de negocio
- One hot encoding de variables categoricas
- Estadisticas descriptivas (tendencia central y dispersion)
- Deteccion de outliers con IQR
- Graficas:
  - Histogramas
  - Dispersion
  - Boxplot de ingresos por segmento
  - Barras de ingresos promedio por region
  - Matriz de correlacion anotada
- Escalamiento con StandardScaler
- Transformacion logaritmica para variables positivas

## Ciclo de vida aplicado
Este trabajo sigue un ciclo de vida de analisis y preparacion de datos orientado a
machine learning:

1. Comprension del negocio:
   - Se busca analizar el rendimiento de empresas fintech por region y segmento.
2. Comprension de los datos:
   - Inspeccion del CSV, tipos de datos, faltantes y estadisticas iniciales.
3. Preparacion de datos:
   - Limpieza, imputacion, validaciones, codificacion y escalamiento.
4. Analisis exploratorio:
   - Graficas y medidas descriptivas para entender distribuciones y relaciones.
5. Preparacion para modelado:
   - Dataset listo para entrenar modelos (variables numericas y codificadas).
6. Evaluacion (siguiente paso):
   - Comparar modelos y metricas cuando se defina el objetivo predictivo.
7. Despliegue (siguiente paso):
   - Guardar modelo y usarlo para predicciones en nuevos datos.