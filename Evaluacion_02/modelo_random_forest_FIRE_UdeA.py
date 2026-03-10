"""
===============================================================================
  MODELO RANDOM FOREST — Clasificación de Riesgo Financiero
  Dataset Sintético Realista — FIRE UdeA
  Variable objetivo: label (0 = sin riesgo, 1 = riesgo financiero)
===============================================================================
"""

# =============================================================================
# 1. IMPORTAR LIBRERÍAS
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'graficas_random_forest'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def guardar_figura(nombre):
    path = os.path.join(OUTPUT_DIR, f'{nombre}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  📁 Gráfica guardada: {path}')

sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.4f}'.format)

print("✅ Librerías importadas correctamente.\n")

# =============================================================================
# 2. CARGAR Y PREPARAR DATOS
# =============================================================================
df = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')
print(f"{'='*70}")
print(f"  DATASET CARGADO: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"{'='*70}\n")

# Distribución de la variable objetivo
print("📊 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (label):")
label_counts = df['label'].value_counts()
for val, count in label_counts.items():
    etiqueta = "Sin riesgo" if val == 0 else "Con riesgo"
    print(f"   {val} ({etiqueta}): {count} ({count/len(df)*100:.1f}%)")

# Columnas numéricas (features)
cols_excluir = ['anio', 'unidad', 'label']
feature_cols = [c for c in df.columns if c not in cols_excluir]
print(f"\n📋 FEATURES SELECCIONADAS ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

# =============================================================================
# 3. TRATAMIENTO DE VALORES NULOS
# =============================================================================
print(f"\n{'='*70}")
print("  TRATAMIENTO DE VALORES NULOS")
print(f"{'='*70}")

nulos_antes = df[feature_cols].isnull().sum()
nulos_antes = nulos_antes[nulos_antes > 0]
if not nulos_antes.empty:
    print("\n  Nulos encontrados:")
    for col, n in nulos_antes.items():
        print(f"   • {col}: {n} nulos ({n/len(df)*100:.1f}%)")

    # Imputación con mediana por unidad
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df.groupby('unidad')[col].transform(
                lambda x: x.fillna(x.median())
            )
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    print(f"\n  ✅ Nulos después de imputar (mediana por unidad): {df[feature_cols].isnull().sum().sum()}")
else:
    print("  ✅ No hay valores nulos en las features.")

# =============================================================================
# 4. SEPARACIÓN DE DATOS (TRAIN / TEST)
# =============================================================================
print(f"\n{'='*70}")
print("  SEPARACIÓN TRAIN / TEST")
print(f"{'='*70}")

X = df[feature_cols]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n  Total:     {len(X)} muestras")
print(f"  Train:     {len(X_train)} muestras ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Test:      {len(X_test)} muestras ({len(X_test)/len(X)*100:.0f}%)")
print(f"\n  Distribución en TRAIN — Sin riesgo: {(y_train==0).sum()}, Con riesgo: {(y_train==1).sum()}")
print(f"  Distribución en TEST  — Sin riesgo: {(y_test==0).sum()}, Con riesgo: {(y_test==1).sum()}")

# Escalar features (para visualizaciones, RF no lo necesita estrictamente)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# =============================================================================
# 5. MODELO BASELINE — Random Forest con parámetros por defecto
# =============================================================================
print(f"\n{'='*70}")
print("  MODELO BASELINE — Random Forest (parámetros por defecto)")
print(f"{'='*70}")

rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
y_pred_base = rf_baseline.predict(X_test)
y_proba_base = rf_baseline.predict_proba(X_test)[:, 1]

print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred_base):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_base):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_base):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred_base):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_base):.4f}")

# =============================================================================
# 6. VALIDACIÓN CRUZADA (BASELINE)
# =============================================================================
print(f"\n{'='*70}")
print("  VALIDACIÓN CRUZADA — Stratified 5-Fold (Baseline)")
print(f"{'='*70}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print(f"\n  {'Métrica':<12} {'Media':>8} {'± Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'-'*48}")
for metric in scoring_metrics:
    scores = cross_val_score(rf_baseline, X, y, cv=cv, scoring=metric)
    print(f"  {metric:<12} {scores.mean():>8.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")

# =============================================================================
# 7. OPTIMIZACIÓN DE HIPERPARÁMETROS — GridSearchCV
# =============================================================================
print(f"\n{'='*70}")
print("  OPTIMIZACIÓN DE HIPERPARÁMETROS — GridSearchCV")
print(f"{'='*70}")

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0,
    refit=True
)
grid_search.fit(X_train, y_train)

print(f"\n  🏆 MEJORES HIPERPARÁMETROS:")
for param, value in grid_search.best_params_.items():
    print(f"     {param}: {value}")
print(f"\n  Mejor F1-Score (CV): {grid_search.best_score_:.4f}")

# =============================================================================
# 8. MODELO OPTIMIZADO — Evaluación Final
# =============================================================================
print(f"\n{'='*70}")
print("  MODELO OPTIMIZADO — Evaluación en Test Set")
print(f"{'='*70}")

rf_best = grid_search.best_estimator_
y_pred = rf_best.predict(X_test)
y_proba = rf_best.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\n  {'Métrica':<12} {'Baseline':>10} {'Optimizado':>12} {'Cambio':>10}")
print(f"  {'-'*46}")
base_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_base),
    'Precision': precision_score(y_test, y_pred_base),
    'Recall': recall_score(y_test, y_pred_base),
    'F1-Score': f1_score(y_test, y_pred_base),
    'ROC-AUC': roc_auc_score(y_test, y_proba_base),
}
opt_metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc}

for name in base_metrics:
    diff = opt_metrics[name] - base_metrics[name]
    signo = "+" if diff >= 0 else ""
    print(f"  {name:<12} {base_metrics[name]:>10.4f} {opt_metrics[name]:>12.4f} {signo}{diff:>9.4f}")

# Classification Report
print(f"\n  📋 REPORTE DE CLASIFICACIÓN (Modelo Optimizado):")
print(classification_report(y_test, y_pred, target_names=['Sin riesgo (0)', 'Con riesgo (1)']))

# Validación cruzada del modelo optimizado
print(f"{'='*70}")
print("  VALIDACIÓN CRUZADA — Stratified 5-Fold (Optimizado)")
print(f"{'='*70}")

print(f"\n  {'Métrica':<12} {'Media':>8} {'± Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'-'*48}")
for metric in scoring_metrics:
    scores = cross_val_score(rf_best, X, y, cv=cv, scoring=metric)
    print(f"  {metric:<12} {scores.mean():>8.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")

# =============================================================================
# 9. GRÁFICA 1 — Matriz de Confusión
# =============================================================================
print(f"\n{'='*70}")
print("  GENERANDO GRÁFICAS...")
print(f"{'='*70}")

cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Matriz de confusión — valores absolutos
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Sin riesgo', 'Con riesgo'],
            yticklabels=['Sin riesgo', 'Con riesgo'], ax=axes[0],
            annot_kws={'size': 18})
axes[0].set_title('Matriz de Confusión (Absoluta)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Valor Real')
axes[0].set_xlabel('Predicción')

# Matriz de confusión — porcentajes
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', cbar=True,
            xticklabels=['Sin riesgo', 'Con riesgo'],
            yticklabels=['Sin riesgo', 'Con riesgo'], ax=axes[1],
            annot_kws={'size': 18}, cbar_kws={'label': '%'})
axes[1].set_title('Matriz de Confusión (Porcentual)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Valor Real')
axes[1].set_xlabel('Predicción')

plt.suptitle('Random Forest — Matriz de Confusión', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('01_matriz_confusion')

# =============================================================================
# 10. GRÁFICA 2 — Curva ROC
# =============================================================================
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)

# También para baseline
fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_base)
roc_auc_base = auc(fpr_base, tpr_base)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'RF Optimizado (AUC = {roc_auc_val:.4f})')
ax.plot(fpr_base, tpr_base, color='#FF9800', lw=2, linestyle='--', label=f'RF Baseline (AUC = {roc_auc_base:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':', label='Aleatorio (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
ax.set_xlabel('False Positive Rate (1 - Especificidad)', fontsize=13)
ax.set_ylabel('True Positive Rate (Sensibilidad)', fontsize=13)
ax.set_title('Curva ROC — Random Forest', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_figura('02_curva_roc')

# =============================================================================
# 11. GRÁFICA 3 — Curva Precision-Recall
# =============================================================================
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(recall_curve, precision_curve, color='#4CAF50', lw=2.5,
        label=f'RF Optimizado (AP = {ap_score:.4f})')
ax.axhline(y=y_test.mean(), color='gray', linestyle=':', lw=1.5,
           label=f'Baseline (prevalencia = {y_test.mean():.4f})')
ax.fill_between(recall_curve, precision_curve, alpha=0.15, color='#4CAF50')
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Curva Precision-Recall — Random Forest', fontsize=16, fontweight='bold')
ax.legend(loc='lower left', fontsize=12)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_figura('03_curva_precision_recall')

# =============================================================================
# 12. GRÁFICA 4 — Importancia de Features
# =============================================================================
importances = rf_best.feature_importances_
std_imp = np.std([tree.feature_importances_ for tree in rf_best.estimators_], axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importancia': importances,
    'Std': std_imp
}).sort_values('Importancia', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Barplot horizontal con barras de error
axes[0].barh(importance_df['Feature'], importance_df['Importancia'],
             xerr=importance_df['Std'], color=sns.color_palette('viridis', len(feature_cols)),
             edgecolor='white', linewidth=0.5, capsize=3)
axes[0].set_xlabel('Importancia (Mean Decrease Impurity)', fontsize=12)
axes[0].set_title('Importancia de Features — Random Forest', fontsize=14, fontweight='bold')
for i, (val, std) in enumerate(zip(importance_df['Importancia'], importance_df['Std'])):
    axes[0].text(val + std + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

# Importancia acumulada
imp_sorted = importance_df.sort_values('Importancia', ascending=False)
imp_sorted['Acumulada'] = imp_sorted['Importancia'].cumsum() / imp_sorted['Importancia'].sum() * 100
axes[1].bar(range(len(imp_sorted)), imp_sorted['Importancia'], color='#42A5F5', alpha=0.7, label='Individual')
ax2 = axes[1].twinx()
ax2.plot(range(len(imp_sorted)), imp_sorted['Acumulada'], color='#EF5350', marker='o',
         linewidth=2, markersize=6, label='Acumulada (%)')
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80%')
ax2.set_ylabel('Importancia acumulada (%)', fontsize=12)
axes[1].set_xticks(range(len(imp_sorted)))
axes[1].set_xticklabels(imp_sorted['Feature'], rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Importancia', fontsize=12)
axes[1].set_title('Importancia Individual y Acumulada (Pareto)', fontsize=14, fontweight='bold')
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

plt.suptitle('Análisis de Importancia de Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('04_importancia_features')

# Top features
print("\n  🏆 TOP FEATURES POR IMPORTANCIA:")
for i, row in imp_sorted.iterrows():
    print(f"     {row['Feature']:<28} {row['Importancia']:.4f}  (acum: {row['Acumulada']:.1f}%)")

# =============================================================================
# 13. GRÁFICA 5 — Distribución de Probabilidades Predichas
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histograma de probabilidades
for label_val, color, nombre in [(0, '#4CAF50', 'Sin riesgo'), (1, '#F44336', 'Con riesgo')]:
    mask = y_test == label_val
    axes[0].hist(y_proba[mask], bins=15, alpha=0.6, color=color, label=nombre, edgecolor='white')
axes[0].axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Umbral (0.5)')
axes[0].set_xlabel('Probabilidad predicha de riesgo', fontsize=12)
axes[0].set_ylabel('Frecuencia', fontsize=12)
axes[0].set_title('Distribución de Probabilidades por Clase Real', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)

# KDE de probabilidades
for label_val, color, nombre in [(0, '#4CAF50', 'Sin riesgo'), (1, '#F44336', 'Con riesgo')]:
    mask = y_test == label_val
    if mask.sum() > 1:
        sns.kdeplot(y_proba[mask], color=color, lw=2.5, label=nombre, ax=axes[1], fill=True, alpha=0.2)
axes[1].axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Umbral (0.5)')
axes[1].set_xlabel('Probabilidad predicha de riesgo', fontsize=12)
axes[1].set_ylabel('Densidad', fontsize=12)
axes[1].set_title('Densidad de Probabilidades por Clase Real (KDE)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.suptitle('Análisis de Probabilidades Predichas', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('05_distribucion_probabilidades')

# =============================================================================
# 14. GRÁFICA 6 — Métricas por Umbral de Decisión
# =============================================================================
thresholds = np.arange(0.05, 0.96, 0.01)
metrics_by_threshold = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    if y_pred_t.sum() == 0 or y_pred_t.sum() == len(y_pred_t):
        continue
    metrics_by_threshold.append({
        'Umbral': t,
        'Accuracy': accuracy_score(y_test, y_pred_t),
        'Precision': precision_score(y_test, y_pred_t, zero_division=0),
        'Recall': recall_score(y_test, y_pred_t, zero_division=0),
        'F1': f1_score(y_test, y_pred_t, zero_division=0),
    })

df_thresholds = pd.DataFrame(metrics_by_threshold)

fig, ax = plt.subplots(figsize=(12, 7))
for metric, color in [('Accuracy', '#2196F3'), ('Precision', '#4CAF50'),
                       ('Recall', '#FF9800'), ('F1', '#9C27B0')]:
    ax.plot(df_thresholds['Umbral'], df_thresholds[metric], lw=2.5, label=metric, color=color)
ax.axvline(x=0.5, color='gray', linestyle='--', lw=1.5, alpha=0.7, label='Umbral default (0.5)')

# Marcar el umbral óptimo (mejor F1)
best_idx = df_thresholds['F1'].idxmax()
best_t = df_thresholds.loc[best_idx, 'Umbral']
best_f1 = df_thresholds.loc[best_idx, 'F1']
ax.axvline(x=best_t, color='red', linestyle=':', lw=2, label=f'Mejor F1 (umbral={best_t:.2f})')
ax.scatter([best_t], [best_f1], color='red', s=100, zorder=5)

ax.set_xlabel('Umbral de Decisión', fontsize=13)
ax.set_ylabel('Métrica', fontsize=13)
ax.set_title('Métricas vs Umbral de Decisión — Random Forest', fontsize=16, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_figura('06_metricas_por_umbral')

print(f"\n  📌 Mejor umbral por F1-Score: {best_t:.2f} (F1 = {best_f1:.4f})")

# =============================================================================
# 15. GRÁFICA 7 — Comparación Baseline vs Optimizado
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

metric_names = list(base_metrics.keys())
base_vals = [base_metrics[m] for m in metric_names]
opt_vals = [opt_metrics[m] for m in metric_names]

x_pos = np.arange(len(metric_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, base_vals, width, label='Baseline', color='#FF9800', alpha=0.8, edgecolor='white')
bars2 = ax.bar(x_pos + width/2, opt_vals, width, label='Optimizado', color='#2196F3', alpha=0.8, edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Comparación: Baseline vs Optimizado — Random Forest', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim([0, 1.15])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
guardar_figura('07_comparacion_baseline_vs_optimizado')

# =============================================================================
# 16. GRÁFICA 8 — Validación Cruzada — Boxplot por Fold
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

cv_results = {}
for metric in scoring_metrics:
    cv_results[metric] = cross_val_score(rf_best, X, y, cv=cv, scoring=metric)

# Boxplot de todas las métricas
cv_df = pd.DataFrame(cv_results)
cv_df.columns = [m.replace('_', ' ').title() for m in scoring_metrics]
bp = cv_df.boxplot(ax=axes[0], grid=False, patch_artist=True, return_type='dict')
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_title('Distribución de Métricas — Validación Cruzada', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].tick_params(axis='x', rotation=30)

# Score por fold
fold_numbers = [f'Fold {i+1}' for i in range(5)]
for metric, color in zip(scoring_metrics, colors):
    axes[1].plot(fold_numbers, cv_results[metric], marker='o', lw=2,
                 label=metric.replace('_', ' ').title(), color=color)
axes[1].set_title('Métricas por Fold — Validación Cruzada', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Análisis de Validación Cruzada (5-Fold)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
guardar_figura('08_validacion_cruzada')

# =============================================================================
# 17. GRÁFICA 9 — Top 2 Features — Frontera de Decisión (aproximada)
# =============================================================================
top2_features = imp_sorted['Feature'].iloc[:2].tolist()
print(f"\n  🔍 Top 2 features para frontera de decisión: {top2_features}")

X_top2 = X[top2_features].values
rf_2d = RandomForestClassifier(**rf_best.get_params(), random_state=42)
rf_2d.fit(X_top2[X_train.index], y.iloc[X_train.index])

# Crear mesh
x_min, x_max = X_top2[:, 0].min() - 0.5 * abs(X_top2[:, 0].std()), X_top2[:, 0].max() + 0.5 * abs(X_top2[:, 0].std())
y_min, y_max = X_top2[:, 1].min() - 0.5 * abs(X_top2[:, 1].std()), X_top2[:, 1].max() + 0.5 * abs(X_top2[:, 1].std())
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = rf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(12, 9))
contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
plt.colorbar(contour, ax=ax, label='P(Riesgo)')

# Puntos de test
scatter = ax.scatter(X_test[top2_features[0]], X_test[top2_features[1]],
                     c=y_test, cmap='RdYlBu_r', edgecolors='black', s=80, linewidth=1.2, zorder=5)
ax.set_xlabel(top2_features[0], fontsize=13)
ax.set_ylabel(top2_features[1], fontsize=13)
ax.set_title(f'Frontera de Decisión (Top 2 features) — Random Forest', fontsize=16, fontweight='bold')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4575B4', markersize=10, label='Sin riesgo (real)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#D73027', markersize=10, label='Con riesgo (real)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
plt.tight_layout()
guardar_figura('09_frontera_decision')

# =============================================================================
# 18. GRÁFICA 10 — Predicciones Individuales en Test
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 6))

test_indices = range(len(y_test))
colors_pred = ['#4CAF50' if p == r else '#F44336' for p, r in zip(y_pred, y_test)]
bars = ax.bar(test_indices, y_proba, color=colors_pred, edgecolor='white', alpha=0.8)
ax.axhline(y=0.5, color='black', linestyle='--', lw=1.5, label='Umbral (0.5)')

# Marcar valores reales
for i, (prob, real) in enumerate(zip(y_proba, y_test)):
    marker = '●' if real == 1 else '○'
    ax.text(i, prob + 0.02, marker, ha='center', fontsize=8)

ax.set_xlabel('Muestra de Test', fontsize=12)
ax.set_ylabel('Probabilidad de Riesgo', fontsize=12)
ax.set_title('Predicciones Individuales en Test — Random Forest', fontsize=16, fontweight='bold')
ax.legend(['Umbral (0.5)', 'Correcto', 'Incorrecto'], fontsize=11, loc='upper right')
ax.set_ylim([-0.02, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
guardar_figura('10_predicciones_individuales')

# =============================================================================
# 19. PARÁMETROS DEL MODELO FINAL
# =============================================================================
print(f"\n{'='*70}")
print("  PARÁMETROS DEL MODELO FINAL")
print(f"{'='*70}")
for param, value in sorted(rf_best.get_params().items()):
    print(f"   {param}: {value}")

# =============================================================================
# 20. RESUMEN FINAL
# =============================================================================
print(f"\n{'='*70}")
print("  RESUMEN FINAL — Random Forest | FIRE UdeA")
print(f"{'='*70}")

print(f"""
  📊 CONFIGURACIÓN:
     • Features: {len(feature_cols)} variables numéricas
     • Train/Test split: 75%/25% (estratificado)
     • Optimización: GridSearchCV con 5-Fold Stratified CV
     • Métrica de optimización: F1-Score

  🏆 RENDIMIENTO DEL MODELO OPTIMIZADO (Test Set):
     • Accuracy:   {acc:.4f}
     • Precision:   {prec:.4f}
     • Recall:      {rec:.4f}
     • F1-Score:    {f1:.4f}
     • ROC-AUC:     {roc:.4f}

  🔑 TOP 3 FEATURES MÁS IMPORTANTES:
     1. {imp_sorted['Feature'].iloc[0]:<28} ({imp_sorted['Importancia'].iloc[0]:.4f})
     2. {imp_sorted['Feature'].iloc[1]:<28} ({imp_sorted['Importancia'].iloc[1]:.4f})
     3. {imp_sorted['Feature'].iloc[2]:<28} ({imp_sorted['Importancia'].iloc[2]:.4f})

  📁 GRÁFICAS GUARDADAS EN: {OUTPUT_DIR}/
     01_matriz_confusion.png
     02_curva_roc.png
     03_curva_precision_recall.png
     04_importancia_features.png
     05_distribucion_probabilidades.png
     06_metricas_por_umbral.png
     07_comparacion_baseline_vs_optimizado.png
     08_validacion_cruzada.png
     09_frontera_decision.png
     10_predicciones_individuales.png
""")

# Guardar métricas en CSV
metrics_export = pd.DataFrame({
    'Metrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Baseline': [base_metrics[m] for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']],
    'Optimizado': [opt_metrics[m] for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
})
metrics_export.to_csv('metricas_random_forest_FIRE_UdeA.csv', index=False)
print("  📁 Métricas exportadas: metricas_random_forest_FIRE_UdeA.csv")

# Guardar importancias en CSV
imp_sorted.to_csv('importancia_features_RF_FIRE_UdeA.csv', index=False)
print("  📁 Importancias exportadas: importancia_features_RF_FIRE_UdeA.csv")

print(f"\n{'='*70}")
print("  FIN DEL MODELADO — Random Forest | FIRE UdeA")
print(f"{'='*70}")
