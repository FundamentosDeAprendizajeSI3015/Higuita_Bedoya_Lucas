# Informe: Evaluacion No Supervisada y Supervisada del Inventario Farmaceutico

## 1. Problema
Se requiere evaluar y mejorar la calidad de etiquetas de categoria en un inventario farmaceutico, considerando que hasta el 30% de las etiquetas puede estar mal asignado.
Luego, se deben entrenar modelos supervisados (Arboles de Decision, Regresion Logistica y Regresion Lineal) y compararlos entre el dataset original y el dataset con etiquetas reevaluadas.

## 2. Dataset y Preparacion
- Registros analizados: 5000
- Variables numericas para clustering: precio_unitario, costo_unitario, margen_unitario, stock_actual, stock_minimo, stock_maximo, demanda_promedio_diaria, desviacion_demanda, ventas_ultimos_30_dias, quiebres_stock_ultimos_6m, lead_time_dias, dias_para_vencer
- Estandarizacion aplicada en variables numericas para analisis de distancias.

## 3. Analisis No Supervisado
Modelos aplicados: K-Means, Fuzzy C-Means, Subtractive Clustering, DBSCAN y clustering jerarquico (familia cluster).

### 3.1 Metricas de Clustering
| metodo                     |   n_clusters |   silhouette |   calinski_harabasz |   davies_bouldin |   noise_rate |
|:---------------------------|-------------:|-------------:|--------------------:|-----------------:|-------------:|
| kmeans                     |           10 |    0.0799408 |             362.668 |          2.56622 |            0 |
| fuzzy_c_means              |            2 |    0.156264  |            1031.16  |          2.16926 |            0 |
| subtractive                |           25 |    0.0410884 |             165.044 |          2.33829 |            0 |
| dbscan                     |            0 |  nan         |             nan     |        nan       |            1 |
| familia_cluster_jerarquico |           10 |    0.0195111 |             256.059 |          3.2134  |            0 |

- Metodo con mejor separacion global (heuristica): fuzzy_c_means.

## 4. Reevaluacion de Etiquetas
Se realizo voto mayoritario entre las etiquetas inferidas por los cinco metodos de clustering.
Solo se cambiaron etiquetas cuando hubo desacuerdo con la etiqueta original y confianza de consenso >= 0.60.
Adicionalmente, se limito el cambio a maximo 30% de los registros. Tasa final de cambio: 22.04%.

## 5. Modelos Supervisados de Clasificacion
Comparacion de Arbol de Decision y Regresion Logistica entrenados con etiquetas originales vs reevaluadas.

| modelo              | dataset_entrenamiento   |   accuracy_vs_proxy |   f1_macro_vs_proxy |   accuracy_vs_original |   f1_macro_vs_original |
|:--------------------|:------------------------|--------------------:|--------------------:|-----------------------:|-----------------------:|
| Arbol_Decision      | original                |              0.9360 |              0.9570 |                 0.7780 |                 0.8056 |
| Arbol_Decision      | reevaluado              |              1.0000 |              1.0000 |                 0.7800 |                 0.8193 |
| Regresion_Logistica | original                |              0.9010 |              0.9218 |                 0.7720 |                 0.7878 |
| Regresion_Logistica | reevaluado              |              1.0000 |              1.0000 |                 0.7800 |                 0.8193 |

Interpretacion clave: si el rendimiento vs proxy (etiqueta reevaluada) sube al entrenar con etiquetas reevaluadas, se evidencia mejora de consistencia en clases.

## 6. Modelo Supervisado de Regresion
Se entreno Regresion Lineal para predecir ventas_ultimos_30_dias comparando dataset original vs reevaluado (modificando la variable de categoria como feature).

| modelo           | dataset_entrenamiento   |     mae |    rmse |     r2 |
|:-----------------|:------------------------|--------:|--------:|-------:|
| Regresion_Lineal | original                | 16.0656 | 19.9179 | 0.9965 |
| Regresion_Lineal | reevaluado              | 16.0589 | 19.9125 | 0.9965 |

## 7. Conclusion
El flujo no supervisado permitio detectar posibles inconsistencias de etiquetado y generar una version reevaluada del dataset.
La comparacion supervisada permite decidir si la reevaluacion aporta mejoras de generalizacion frente al dataset original.
Se recomienda usar la version reevaluada cuando mejore de forma consistente accuracy/F1 (clasificacion) y/o MAE-RMSE-R2 (regresion).

## 8. Graficas Generadas
Se generaron las siguientes visualizaciones en la carpeta graficas_informe:
- graficas_informe/01_distribucion_categorias_original.png
- graficas_informe/02_distribucion_categorias_reevaluada.png
- graficas_informe/03_cambios_por_categoria.png
- graficas_informe/04_histograma_confianza_consenso.png
- graficas_informe/05_heatmap_correlaciones.png
- graficas_informe/06_metricas_clustering_normalizadas.png
- graficas_informe/07_pca_clusters_kmeans.png
- graficas_informe/08_pca_clusters_fuzzy_c_means.png
- graficas_informe/09_pca_clusters_subtractive.png
- graficas_informe/10_pca_clusters_dbscan.png
- graficas_informe/11_pca_clusters_familia_cluster_jerarquico.png
- graficas_informe/12_boxplot_ventas_por_categoria_reevaluada.png
- graficas_informe/13_accuracy_clasificacion.png
- graficas_informe/14_f1_clasificacion.png
- graficas_informe/15_metricas_regresion_lineal.png
- graficas_informe/16_scatter_precio_vs_demanda.png