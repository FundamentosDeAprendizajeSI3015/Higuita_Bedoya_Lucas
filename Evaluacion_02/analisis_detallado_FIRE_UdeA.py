"""
===============================================================================
  ANÁLISIS EXPLORATORIO DE DATOS (EDA) DETALLADO
  Dataset Sintético Realista — FIRE UdeA
  Indicadores Financieros de Riesgo — Universidad de Antioquia (2016–2025)
===============================================================================

Variables del dataset:
  - anio: Año fiscal (2016–2025)
  - unidad: Unidad académica/administrativa
  - ingresos_totales, gastos_personal: En COP
  - liquidez, dias_efectivo, cfo: Indicadores de liquidez
  - participacion_ley30, regalias, servicios, matriculas: Fuentes de ingreso
  - hhi_fuentes: Índice Herfindahl-Hirschman de concentración
  - endeudamiento, tendencia_ingresos, gp_ratio: Ratios financieros
  - label: Variable objetivo (0 = sin riesgo, 1 = riesgo financiero)
"""

# =============================================================================
# 1. IMPORTAR LIBRERÍAS
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, pointbiserialr, ttest_ind, f_oneway, chi2_contingency
import warnings

warnings.filterwarnings('ignore')

# Crear carpeta para guardar gráficas
OUTPUT_DIR = 'graficas_FIRE_UdeA'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def guardar_figura(nombre):
    """Guarda la figura actual en la carpeta de salida y cierra."""
    path = os.path.join(OUTPUT_DIR, f'{nombre}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  📁 Gráfica guardada: {path}')

sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.4f}'.format)

print("✅ Librerías importadas correctamente.\n")

# =============================================================================
# 2. CARGAR Y EXPLORAR EL DATASET
# =============================================================================
df = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')

print(f"{'='*60}")
print(f"  DIMENSIONES DEL DATASET: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"{'='*60}\n")

print("📋 COLUMNAS DEL DATASET:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n{'='*60}")
print("  PRIMERAS 5 FILAS")
print(f"{'='*60}")
print(df.head().to_string())

print(f"\n{'='*60}")
print("  ÚLTIMAS 5 FILAS")
print(f"{'='*60}")
print(df.tail().to_string())

print(f"\n{'='*60}")
print("  INFORMACIÓN GENERAL DEL DATASET")
print(f"{'='*60}")
df.info()

# =============================================================================
# 3. INSPECCIÓN DE TIPOS DE DATOS Y VALORES NULOS
# =============================================================================
print(f"\n{'='*60}")
print("  TIPOS DE DATOS POR COLUMNA")
print(f"{'='*60}")
print(df.dtypes.to_frame('Tipo').to_string())

print(f"\n{'='*60}")
print("  ANÁLISIS DE VALORES NULOS")
print(f"{'='*60}")
null_analysis = pd.DataFrame({
    'Nulos': df.isnull().sum(),
    'No Nulos': df.notnull().sum(),
    '% Nulos': (df.isnull().sum() / len(df) * 100).round(2)
})
null_analysis = null_analysis.sort_values('% Nulos', ascending=False)
print(null_analysis.to_string())
total_nulls = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"\nTotal de celdas con valores nulos: {total_nulls} de {total_cells} ({total_nulls/total_cells*100:.2f}%)")

# Heatmap de valores nulos
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(df.isnull().T, cbar=True, cmap='YlOrRd', yticklabels=df.columns,
            xticklabels=False, cbar_kws={'label': 'Nulo (1) / No Nulo (0)'})
ax.set_title('Mapa de Calor de Valores Nulos por Variable', fontsize=16, fontweight='bold')
ax.set_ylabel('Variables')
ax.set_xlabel(f'Observaciones (n={len(df)})')
plt.tight_layout()
guardar_figura('01_heatmap_valores_nulos')

# Nulos por unidad
print("\n📊 NULOS POR UNIDAD ACADÉMICA:")
null_by_unit = df.groupby('unidad').apply(lambda x: x.isnull().sum().sum())
for unit, count in null_by_unit.items():
    print(f"  {unit}: {count} valores nulos")

# =============================================================================
# 4. ESTADÍSTICAS DESCRIPTIVAS COMPLETAS
# =============================================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_no_label = [c for c in numeric_cols if c not in ['label', 'anio']]

print(f"\n{'='*60}")
print("  ESTADÍSTICAS DESCRIPTIVAS — VARIABLES NUMÉRICAS")
print(f"{'='*60}")
print(df[numeric_cols].describe().T.to_string())

# Métricas avanzadas
print(f"\n{'='*60}")
print("  MÉTRICAS AVANZADAS")
print(f"{'='*60}")
advanced_stats = pd.DataFrame({
    'Media': df[numeric_cols_no_label].mean(),
    'Mediana': df[numeric_cols_no_label].median(),
    'Desv. Est.': df[numeric_cols_no_label].std(),
    'Asimetría (Skew)': df[numeric_cols_no_label].skew(),
    'Curtosis': df[numeric_cols_no_label].kurtosis(),
    'Coef. Variación (%)': (df[numeric_cols_no_label].std() / df[numeric_cols_no_label].mean() * 100).round(2),
    'IQR': df[numeric_cols_no_label].quantile(0.75) - df[numeric_cols_no_label].quantile(0.25),
    'Rango': df[numeric_cols_no_label].max() - df[numeric_cols_no_label].min(),
    'Mín': df[numeric_cols_no_label].min(),
    'Máx': df[numeric_cols_no_label].max()
})
print(advanced_stats.round(4).to_string())

# Estadísticas de variable categórica
print(f"\n{'='*60}")
print("  ESTADÍSTICAS — VARIABLE CATEGÓRICA (unidad)")
print(f"{'='*60}")
print(f"\nUnidades únicas: {df['unidad'].nunique()}")
print(f"\nFrecuencia absoluta y relativa:")
freq = df['unidad'].value_counts()
freq_rel = df['unidad'].value_counts(normalize=True) * 100
cat_stats = pd.DataFrame({'Frecuencia': freq, 'Porcentaje (%)': freq_rel.round(2)})
print(cat_stats.to_string())

# Distribución de label
print(f"\n{'='*60}")
print("  DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (label)")
print(f"{'='*60}")
label_counts = df['label'].value_counts()
print(f"  Sin riesgo (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
print(f"  Con riesgo (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"  Ratio riesgo/total: {label_counts.get(1, 0)/len(df):.3f}")

# =============================================================================
# 5. TRATAMIENTO DE VALORES NULOS Y DUPLICADOS
# =============================================================================
n_duplicados = df.duplicated().sum()
print(f"\n🔍 Filas duplicadas encontradas: {n_duplicados}")
if n_duplicados > 0:
    df = df.drop_duplicates()
    print(f"   Se eliminaron {n_duplicados} duplicados. Nuevo shape: {df.shape}")

print(f"\n{'='*60}")
print("  ANTES DEL TRATAMIENTO DE NULOS")
print(f"{'='*60}")
print(f"  Valores nulos totales: {df.isnull().sum().sum()}")

df_original = df.copy()

# Imputación con mediana por unidad académica
for col in numeric_cols_no_label:
    if df[col].isnull().any():
        df[col] = df.groupby('unidad')[col].transform(
            lambda x: x.fillna(x.median())
        )
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

print(f"\n{'='*60}")
print("  DESPUÉS DEL TRATAMIENTO DE NULOS")
print(f"{'='*60}")
print(f"  Valores nulos totales: {df.isnull().sum().sum()}")

comparison = pd.DataFrame({
    'Nulos ANTES': df_original.isnull().sum(),
    'Nulos DESPUÉS': df.isnull().sum()
})
comparison = comparison[comparison['Nulos ANTES'] > 0]
if not comparison.empty:
    print("\n📊 Columnas imputadas:")
    print(comparison.to_string())

# =============================================================================
# 6. ANÁLISIS DE DISTRIBUCIÓN DE VARIABLES NUMÉRICAS
# =============================================================================
n_vars = len(numeric_cols_no_label)
n_cols_plot = 4
n_rows_plot = (n_vars + n_cols_plot - 1) // n_cols_plot

# Histogramas + KDE
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(20, 4 * n_rows_plot))
axes = axes.flatten()

for i, col in enumerate(numeric_cols_no_label):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=15, density=True, alpha=0.6, color='steelblue', edgecolor='white')
    if len(data) > 2:
        data.plot.kde(ax=ax, color='darkred', linewidth=2)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Densidad')
    ax.axvline(data.mean(), color='green', linestyle='--', label=f'Media: {data.mean():.2f}')
    ax.axvline(data.median(), color='orange', linestyle=':', label=f'Mediana: {data.median():.2f}')
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Distribución de Variables Numéricas (Histograma + KDE)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('02_histogramas_kde')

# Gráficos Q-Q y test de Shapiro-Wilk
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(20, 4 * n_rows_plot))
axes = axes.flatten()

shapiro_results = {}
for i, col in enumerate(numeric_cols_no_label):
    ax = axes[i]
    data = df[col].dropna()
    stats.probplot(data, dist="norm", plot=ax)

    if len(data) >= 3:
        stat_sw, p_val = shapiro(data)
        shapiro_results[col] = {'Estadístico W': stat_sw, 'p-value': p_val,
                                'Normal (α=0.05)': '✅ Sí' if p_val > 0.05 else '❌ No'}
        color = 'green' if p_val > 0.05 else 'red'
        ax.set_title(f'{col}\nShapiro p={p_val:.4f}', fontsize=10, fontweight='bold', color=color)
    else:
        ax.set_title(col, fontsize=10)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Gráficos Q-Q para Evaluación de Normalidad', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('03_qq_plots_normalidad')

print(f"\n{'='*60}")
print("  TEST DE SHAPIRO-WILK — RESULTADOS DE NORMALIDAD")
print(f"{'='*60}")
shapiro_df = pd.DataFrame(shapiro_results).T
print(shapiro_df.to_string())

# =============================================================================
# 7. ANÁLISIS DE VARIABLES CATEGÓRICAS
# =============================================================================

# Distribución por unidad académica
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

order = df['unidad'].value_counts().index
sns.countplot(data=df, y='unidad', order=order, ax=axes[0], palette='Set2')
axes[0].set_title('Frecuencia por Unidad Académica', fontweight='bold')
axes[0].set_xlabel('Número de Observaciones')

colors = sns.color_palette('Set2', n_colors=df['unidad'].nunique())
df['unidad'].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', colors=colors,
                                       startangle=90, textprops={'fontsize': 9})
axes[1].set_title('Proporción por Unidad Académica', fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
guardar_figura('04_distribucion_unidades')

# Distribución de label por unidad
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
ct = pd.crosstab(df['unidad'], df['label'], normalize='index') * 100
ct.columns = ['Sin Riesgo (%)', 'Con Riesgo (%)']

ct.plot(kind='barh', stacked=True, ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Proporción de Riesgo Financiero por Unidad', fontweight='bold')
axes[0].set_xlabel('Porcentaje (%)')
axes[0].legend(loc='lower right')

sns.countplot(data=df, x='label', palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_title('Distribución de la Variable Objetivo (label)', fontweight='bold')
axes[1].set_xlabel('Label (0=Sin Riesgo, 1=Riesgo)')
axes[1].set_ylabel('Frecuencia')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.tight_layout()
guardar_figura('05_label_por_unidad')

# Boxplots agrupados: variables clave por label
key_vars = ['ingresos_totales', 'gastos_personal', 'liquidez', 'endeudamiento', 'gp_ratio', 'cfo']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, var in enumerate(key_vars):
    sns.boxplot(data=df, x='label', y=var, hue='label', palette=['#2ecc71', '#e74c3c'], ax=axes[i], legend=False)
    axes[i].set_title(f'{var} por Label', fontweight='bold')
    axes[i].set_xlabel('Label (0=Sin Riesgo, 1=Riesgo)')

fig.suptitle('Variables Financieras Clave según Riesgo Financiero', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
guardar_figura('06_boxplots_por_label')

# =============================================================================
# 8. DETECCIÓN Y TRATAMIENTO DE OUTLIERS
# =============================================================================
outlier_summary = []

for col in numeric_cols_no_label:
    data = df[col].dropna()

    # Método IQR
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    lower_iqr, upper_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers_iqr = ((data < lower_iqr) | (data > upper_iqr)).sum()

    # Método Z-Score
    z_scores = np.abs(stats.zscore(data))
    n_outliers_zscore = (z_scores > 3).sum()

    outlier_summary.append({
        'Variable': col,
        'Outliers IQR': n_outliers_iqr,
        '% Outliers IQR': round(n_outliers_iqr / len(data) * 100, 2),
        'Outliers Z-Score': n_outliers_zscore,
        '% Outliers Z-Score': round(n_outliers_zscore / len(data) * 100, 2),
        'Límite Inf IQR': lower_iqr,
        'Límite Sup IQR': upper_iqr
    })

outlier_df = pd.DataFrame(outlier_summary)
print(f"\n{'='*60}")
print("  DETECCIÓN DE OUTLIERS — Resumen")
print(f"{'='*60}")
print(outlier_df.to_string(index=False))

# Boxplots individuales
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(20, 4 * n_rows_plot))
axes = axes.flatten()

for i, col in enumerate(numeric_cols_no_label):
    sns.boxplot(data=df, y=col, ax=axes[i], color='steelblue', width=0.4)
    axes[i].set_title(col, fontsize=11, fontweight='bold')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Boxplots — Visualización de Outliers por Variable', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('07_boxplots_outliers')

# Winsorización (capping) P5-P95
df_winsorized = df.copy()
cols_with_outliers = outlier_df[outlier_df['Outliers IQR'] > 0]['Variable'].tolist()

print(f"\n🔧 Aplicando winsorización (P5-P95) a: {cols_with_outliers}")
for col in cols_with_outliers:
    p5, p95 = df[col].quantile(0.05), df[col].quantile(0.95)
    df_winsorized[col] = df[col].clip(lower=p5, upper=p95)

print("✅ Winsorización aplicada (copia separada). Dataset original preservado.")

# =============================================================================
# 9. MATRIZ DE CORRELACIÓN Y ANÁLISIS BIVARIADO
# =============================================================================
corr_cols = numeric_cols_no_label + ['label']
corr_pearson = df[corr_cols].corr(method='pearson')

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Pearson
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
sns.heatmap(corr_pearson, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=axes[0], square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
axes[0].set_title('Correlación de Pearson', fontsize=14, fontweight='bold')

# Spearman
corr_spearman = df[corr_cols].corr(method='spearman')
mask2 = np.triu(np.ones_like(corr_spearman, dtype=bool))
sns.heatmap(corr_spearman, mask=mask2, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=axes[1], square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
axes[1].set_title('Correlación de Spearman', fontsize=14, fontweight='bold')

fig.suptitle('Matrices de Correlación', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('08_matrices_correlacion')

# Top correlaciones con label
print(f"\n{'='*60}")
print("  TOP CORRELACIONES CON label (Pearson)")
print(f"{'='*60}")
corr_with_label = corr_pearson['label'].drop('label').sort_values(key=abs, ascending=False)
for var, corr_val in corr_with_label.items():
    sign = "+" if corr_val > 0 else ""
    print(f"  {var:30s} → {sign}{corr_val:.4f}")

# Correlación punto-biserial
print(f"\n{'='*60}")
print("  CORRELACIÓN PUNTO-BISERIAL (label vs. variables numéricas)")
print(f"{'='*60}")
pb_results = []
for col in numeric_cols_no_label:
    clean = df[[col, 'label']].dropna()
    if len(clean) > 2:
        corr_pb, p_val_pb = pointbiserialr(clean['label'], clean[col])
        pb_results.append({'Variable': col, 'r_pb': round(corr_pb, 4), 'p-value': round(p_val_pb, 4),
                           'Significativa': '✅' if p_val_pb < 0.05 else '❌'})
pb_df = pd.DataFrame(pb_results).sort_values('p-value')
print(pb_df.to_string(index=False))

# Scatter plots de los 6 pares con mayor correlación
top_pairs = []
for i_idx in range(len(corr_cols)):
    for j_idx in range(i_idx + 1, len(corr_cols)):
        c1, c2 = corr_cols[i_idx], corr_cols[j_idx]
        if c1 != 'label' and c2 != 'label':
            top_pairs.append((c1, c2, abs(corr_pearson.loc[c1, c2])))
top_pairs.sort(key=lambda x: x[2], reverse=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()
for i, (v1, v2, r) in enumerate(top_pairs[:6]):
    sns.scatterplot(data=df, x=v1, y=v2, hue='label', palette=['#2ecc71', '#e74c3c'],
                    ax=axes[i], alpha=0.7, s=80, edgecolor='white')
    axes[i].set_title(f'{v1} vs {v2}\nr = {r:.3f}', fontsize=10, fontweight='bold')
    axes[i].legend(title='Label')
fig.suptitle('Top 6 Pares con Mayor Correlación (Pearson)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('09_scatter_top_correlaciones')

# =============================================================================
# 10. ANÁLISIS DE TENDENCIAS TEMPORALES
# =============================================================================
temporal_vars = ['ingresos_totales', 'gastos_personal', 'liquidez', 'endeudamiento', 'gp_ratio', 'cfo']

fig, axes = plt.subplots(3, 2, figsize=(20, 16))
axes = axes.flatten()

for i, var in enumerate(temporal_vars):
    for unidad in df['unidad'].unique():
        data_u = df[df['unidad'] == unidad].sort_values('anio')
        axes[i].plot(data_u['anio'], data_u[var], marker='o', markersize=4, label=unidad, alpha=0.8)
    axes[i].set_title(f'Evolución de {var}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Año')
    axes[i].set_ylabel(var)
    axes[i].legend(fontsize=7, loc='best')
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Tendencias Temporales por Unidad Académica (2016–2025)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
guardar_figura('10_tendencias_temporales')

# Promedios anuales y tasas de crecimiento
agg_anual = df.groupby('anio')[['ingresos_totales', 'gastos_personal', 'gp_ratio', 'endeudamiento']].mean()
agg_anual['crecimiento_ingresos_%'] = agg_anual['ingresos_totales'].pct_change() * 100
agg_anual['crecimiento_gastos_%'] = agg_anual['gastos_personal'].pct_change() * 100

print(f"\n{'='*60}")
print("  PROMEDIOS ANUALES (TODAS LAS UNIDADES)")
print(f"{'='*60}")
print(agg_anual.round(4).to_string())

# Promedio móvil y % riesgo por año
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for unidad in df['unidad'].unique():
    data_u = df[df['unidad'] == unidad].sort_values('anio')
    ma3 = data_u['ingresos_totales'].rolling(window=3, min_periods=1).mean()
    axes[0].plot(data_u['anio'], ma3, marker='s', markersize=4, label=unidad, alpha=0.8)
axes[0].set_title('Promedio Móvil (3 años) — Ingresos Totales', fontweight='bold')
axes[0].set_xlabel('Año')
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

risk_by_year = df.groupby('anio')['label'].mean() * 100
axes[1].bar(risk_by_year.index, risk_by_year.values, color='#e74c3c', alpha=0.7, edgecolor='white')
axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
axes[1].set_title('% de Unidades en Riesgo por Año', fontweight='bold')
axes[1].set_xlabel('Año')
axes[1].set_ylabel('% en Riesgo')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
guardar_figura('11_promedio_movil_riesgo_anual')

# =============================================================================
# 11. SEGMENTACIÓN Y AGRUPACIÓN DE DATOS
# =============================================================================
print(f"\n{'='*60}")
print("  MÉTRICAS AGREGADAS POR UNIDAD ACADÉMICA")
print(f"{'='*60}")

agg_unidad = df.groupby('unidad').agg({
    'ingresos_totales': ['mean', 'median', 'std', 'min', 'max'],
    'gastos_personal': ['mean', 'median'],
    'liquidez': ['mean', 'std'],
    'endeudamiento': ['mean', 'std'],
    'gp_ratio': ['mean', 'std'],
    'label': ['sum', 'mean']
}).round(4)
agg_unidad.columns = ['_'.join(col) for col in agg_unidad.columns]
print(agg_unidad.to_string())

# Tabla pivote gp_ratio
print(f"\n{'='*60}")
print("  TABLA PIVOTE: gp_ratio (Año × Unidad)")
print(f"{'='*60}")
pivot_gp = df.pivot_table(values='gp_ratio', index='anio', columns='unidad', aggfunc='mean')
print(pivot_gp.round(4).to_string())

# Heatmaps de pivotes
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

sns.heatmap(pivot_gp, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0], linewidths=0.5)
axes[0].set_title('gp_ratio por Año y Unidad', fontsize=13, fontweight='bold')

pivot_end = df.pivot_table(values='endeudamiento', index='anio', columns='unidad', aggfunc='mean')
sns.heatmap(pivot_end, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1], linewidths=0.5)
axes[1].set_title('Endeudamiento por Año y Unidad', fontsize=13, fontweight='bold')

plt.tight_layout()
guardar_figura('12_heatmaps_gpratio_endeudamiento')

# Composición de fuentes de ingreso por unidad
fuentes = ['participacion_ley30', 'participacion_regalias', 'participacion_servicios', 'participacion_matriculas']
fuentes_agg = df.groupby('unidad')[fuentes].mean()

fig, ax = plt.subplots(figsize=(14, 7))
fuentes_agg.plot(kind='barh', stacked=True, ax=ax,
                 color=['#3498db', '#e67e22', '#2ecc71', '#9b59b6'])
ax.set_title('Composición Promedio de Fuentes de Ingreso por Unidad', fontsize=14, fontweight='bold')
ax.set_xlabel('Participación Promedio')
ax.legend(title='Fuente', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
guardar_figura('13_composicion_fuentes_ingreso')

# =============================================================================
# 12. VISUALIZACIONES AVANZADAS — PAIRPLOT Y CLUSTERMAP
# =============================================================================

# Pairplot
pairplot_vars = ['liquidez', 'endeudamiento', 'gp_ratio', 'tendencia_ingresos', 'hhi_fuentes', 'label']
g = sns.pairplot(df[pairplot_vars], hue='label', palette=['#2ecc71', '#e74c3c'],
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'white'},
                 height=2.5)
g.figure.suptitle('Pairplot — Variables Clave por Riesgo Financiero', fontsize=16, fontweight='bold', y=1.02)
guardar_figura('14_pairplot')

# Clustermap jerárquico
cluster_data = df[numeric_cols_no_label].apply(lambda x: (x - x.mean()) / x.std())
g = sns.clustermap(cluster_data.T, cmap='RdBu_r', figsize=(16, 10), linewidths=0.3,
                   method='ward', metric='euclidean',
                   col_colors=[['#2ecc71' if l == 0 else '#e74c3c' for l in df['label']]],
                   cbar_kws={'label': 'Z-Score'})
g.figure.suptitle('Clustermap Jerárquico — Variables Financieras Estandarizadas',
                   fontsize=14, fontweight='bold', y=1.02)
guardar_figura('15_clustermap_jerarquico')

# Dashboard resumen 3x3
fig, axes = plt.subplots(3, 3, figsize=(22, 18))

sns.countplot(data=df, x='label', palette=['#2ecc71', '#e74c3c'], ax=axes[0, 0])
axes[0, 0].set_title('Distribución de Label', fontweight='bold')

df.groupby('unidad')['ingresos_totales'].mean().sort_values().plot(kind='barh', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title('Ingresos Promedio por Unidad', fontweight='bold')

sns.violinplot(data=df, x='label', y='gp_ratio', palette=['#2ecc71', '#e74c3c'], ax=axes[0, 2])
axes[0, 2].set_title('gp_ratio por Label', fontweight='bold')

sns.violinplot(data=df, x='label', y='endeudamiento', palette=['#2ecc71', '#e74c3c'], ax=axes[1, 0])
axes[1, 0].set_title('Endeudamiento por Label', fontweight='bold')

sns.violinplot(data=df, x='label', y='liquidez', palette=['#2ecc71', '#e74c3c'], ax=axes[1, 1])
axes[1, 1].set_title('Liquidez por Label', fontweight='bold')

sns.violinplot(data=df, x='label', y='hhi_fuentes', palette=['#2ecc71', '#e74c3c'], ax=axes[1, 2])
axes[1, 2].set_title('HHI Fuentes por Label', fontweight='bold')

df.groupby('anio')['ingresos_totales'].mean().plot(ax=axes[2, 0], marker='o', color='steelblue')
axes[2, 0].set_title('Ingresos Promedio por Año', fontweight='bold')

corr_label_top = corr_with_label.head(5)
corr_label_top.plot(kind='barh', ax=axes[2, 1], color=['#e74c3c' if v > 0 else '#3498db' for v in corr_label_top])
axes[2, 1].set_title('Top 5 Correlaciones con Label', fontweight='bold')

risk_by_year.plot(kind='bar', ax=axes[2, 2], color='#e74c3c', alpha=0.7)
axes[2, 2].set_title('% Riesgo Financiero por Año', fontweight='bold')
axes[2, 2].set_ylabel('%')

fig.suptitle('DASHBOARD RESUMEN — FIRE UdeA', fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
guardar_figura('16_dashboard_resumen')

# =============================================================================
# 13. PRUEBAS ESTADÍSTICAS
# =============================================================================

# --- Prueba t de Student ---
print(f"\n{'='*70}")
print("  PRUEBA T DE STUDENT — Comparación de Medias (label=0 vs label=1)")
print(f"{'='*70}")

grupo_0 = df[df['label'] == 0]
grupo_1 = df[df['label'] == 1]

ttest_results = []
for col in numeric_cols_no_label:
    g0 = grupo_0[col].dropna()
    g1 = grupo_1[col].dropna()
    if len(g0) > 1 and len(g1) > 1:
        t_stat, p_val = ttest_ind(g0, g1, equal_var=False)
        ttest_results.append({
            'Variable': col,
            'Media (Sin Riesgo)': g0.mean(),
            'Media (Riesgo)': g1.mean(),
            'Diferencia': g1.mean() - g0.mean(),
            't-stat': round(t_stat, 4),
            'p-value': round(p_val, 4),
            'Significativa (α=0.05)': '✅ Sí' if p_val < 0.05 else '❌ No'
        })

ttest_df = pd.DataFrame(ttest_results).sort_values('p-value')
print(ttest_df.to_string(index=False))

# --- ANOVA de una vía ---
print(f"\n{'='*70}")
print("  ANOVA DE UNA VÍA — Comparación entre Unidades Académicas")
print(f"{'='*70}")

anova_results = []
for col in numeric_cols_no_label:
    groups = [group[col].dropna().values for _, group in df.groupby('unidad') if len(group[col].dropna()) > 0]
    if len(groups) > 1:
        f_stat, p_val = f_oneway(*groups)
        anova_results.append({
            'Variable': col,
            'F-stat': round(f_stat, 4),
            'p-value': round(p_val, 4),
            'Significativa (α=0.05)': '✅ Sí' if p_val < 0.05 else '❌ No'
        })

anova_df = pd.DataFrame(anova_results).sort_values('p-value')
print(anova_df.to_string(index=False))

# --- Prueba Chi-Cuadrado ---
print(f"\n{'='*70}")
print("  PRUEBA CHI-CUADRADO — Independencia entre Unidad y Label")
print(f"{'='*70}")

contingency = pd.crosstab(df['unidad'], df['label'])
print("\nTabla de Contingencia:")
print(contingency.to_string())

chi2, p_val_chi, dof, expected = chi2_contingency(contingency)
print(f"\n  χ² = {chi2:.4f}")
print(f"  Grados de libertad = {dof}")
print(f"  p-value = {p_val_chi:.4f}")
print(f"  Resultado: {'Dependientes ✅ (se rechaza H₀)' if p_val_chi < 0.05 else 'Independientes ❌ (no se rechaza H₀)'}")
print(f"\n  Interpretación: {'Existe asociación significativa' if p_val_chi < 0.05 else 'No hay evidencia de asociación significativa'} entre la unidad académica y el riesgo financiero al nivel α=0.05")

print("\n  Frecuencias esperadas bajo H₀:")
print(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).round(2).to_string())

# =============================================================================
# 14. RESUMEN DE HALLAZGOS CLAVE
# =============================================================================
print("\n" + "=" * 70)
print("      RESUMEN DE HALLAZGOS CLAVE — FIRE UdeA Dataset Realista")
print("=" * 70)

total_obs = len(df)
n_unidades = df['unidad'].nunique()
n_anios = df['anio'].nunique()
pct_riesgo = df['label'].mean() * 100

print(f"\n📊 DIMENSIONES:")
print(f"   • {total_obs} observaciones | {n_unidades} unidades | {n_anios} años (2016-2025)")
print(f"   • {len(numeric_cols_no_label)} variables numéricas + 1 categórica + 1 objetivo")

print(f"\n⚠️  RIESGO FINANCIERO:")
print(f"   • {pct_riesgo:.1f}% de las observaciones clasificadas con riesgo financiero")
print(f"   • Sin riesgo: {(df['label']==0).sum()} | Con riesgo: {(df['label']==1).sum()}")

risk_rate = df.groupby('unidad')['label'].mean().sort_values(ascending=False)
print(f"   • Unidad con MAYOR riesgo: {risk_rate.index[0]} ({risk_rate.iloc[0]*100:.1f}%)")
print(f"   • Unidad con MENOR riesgo: {risk_rate.index[-1]} ({risk_rate.iloc[-1]*100:.1f}%)")

print(f"\n💰 INDICADORES FINANCIEROS CLAVE:")
print(f"   • Ingresos totales promedio: ${df['ingresos_totales'].mean():,.0f} COP")
print(f"   • Gastos de personal promedio: ${df['gastos_personal'].mean():,.0f} COP")
print(f"   • GP Ratio promedio: {df['gp_ratio'].mean():.4f}")
print(f"   • Liquidez promedio: {df['liquidez'].mean():.4f}")
print(f"   • Endeudamiento promedio: {df['endeudamiento'].mean():.4f}")

print(f"\n🔗 CORRELACIONES SIGNIFICATIVAS CON RIESGO:")
sig_vars = pb_df[pb_df['Significativa'] == '✅']
if not sig_vars.empty:
    for _, row in sig_vars.iterrows():
        direction = "↑ positiva" if row['r_pb'] > 0 else "↓ negativa"
        print(f"   • {row['Variable']}: r={row['r_pb']:.4f} ({direction}, p={row['p-value']:.4f})")
else:
    print("   • Ninguna variable muestra correlación punto-biserial significativa con label (α=0.05)")

sig_ttest = ttest_df[ttest_df['Significativa (α=0.05)'] == '✅ Sí']
print(f"\n📈 PRUEBAS T SIGNIFICATIVAS (label 0 vs 1):")
if not sig_ttest.empty:
    for _, row in sig_ttest.iterrows():
        print(f"   • {row['Variable']}: p={row['p-value']:.4f} (diferencia de medias = {row['Diferencia']:.4f})")
else:
    print("   • Ninguna variable muestra diferencias significativas de medias entre grupos de riesgo")

sig_anova = anova_df[anova_df['Significativa (α=0.05)'] == '✅ Sí']
print(f"\n🏛️  ANOVA — Variables que difieren significativamente entre unidades:")
if not sig_anova.empty:
    for _, row in sig_anova.iterrows():
        print(f"   • {row['Variable']}: F={row['F-stat']:.4f}, p={row['p-value']:.4f}")
else:
    print("   • Ninguna variable difiere significativamente entre unidades")

print(f"\n📉 VALORES NULOS:")
print(f"   • Total de valores nulos en el dataset original: {df_original.isnull().sum().sum()}")
null_cols = df_original.isnull().sum()
null_cols = null_cols[null_cols > 0]
if not null_cols.empty:
    print(f"   • Columnas afectadas: {', '.join(null_cols.index.tolist())}")
    print(f"   • Estrategia: Imputación con mediana por unidad académica")

print(f"\n{'='*70}")
print("      FIN DEL ANÁLISIS EXPLORATORIO")
print(f"{'='*70}")
