# iris_analysis.py
# -*- coding: utf-8 -*-
"""
Análisis completo del dataset Iris en Python.
Incluye: carga de datos, exploración, visualización, ingeniería de características,
preprocesamiento, partición, entrenamiento de múltiples modelos, evaluación,
validación cruzada, ajuste de hiperparámetros, interpretación y guardado del mejor modelo.

Para ejecutar:
    python iris_analysis.py

Requisitos:
    pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==========================================
# 1) Carga de datos
# ==========================================
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")
class_names = iris.target_names

# Renombrar columnas a formato limpio
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]

# Combinar en un solo DataFrame para EDA
df = pd.concat([X, y.map({i: name for i, name in enumerate(class_names)}).rename("species")], axis=1)

# Crear carpeta de salidas
os.makedirs("outputs", exist_ok=True)

# ==========================================
# 2) Exploración inicial
# ==========================================
print("\n=== Dimensiones ===")
print(df.shape)

print("\n=== Primeras filas ===")
print(df.head())

print("\n=== Descripción estadística ===")
print(df.describe(include="all"))

print("\n=== Clases ===")
print(df["species"].value_counts())

# Correlación
corr = df.drop(columns=["species"]).corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
plt.title("Matriz de correlación - Iris")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

# Pairplot por especie
sns.pairplot(df, hue="species", corner=True)
plt.suptitle("Pairplot Iris", y=1.02)
plt.savefig("outputs/pairplot.png", dpi=150)
plt.close()

# Boxplots
plt.figure(figsize=(10,6))
for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x="species", y=col)
    plt.xlabel("")
    plt.ylabel(col)
plt.suptitle("Distribuciones por especie")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("outputs/boxplots.png", dpi=150)
plt.close()

# ==========================================
# 3) Ingeniería de características (opcionales)
# ==========================================
# Razones y productos simples
X_feat = X.copy()
X_feat["sepal_ratio"] = X["sepal_length"]/X["sepal_width"]
X_feat["petal_ratio"] = X["petal_length"]/X["petal_width"]
X_feat["sepal_area"] = X["sepal_length"]*X["sepal_width"]
X_feat["petal_area"] = X["petal_length"]*X["petal_width"]

# PCA (solo para visualización / diagnóstico)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
plt.figure(figsize=(6,5))
for i, name in enumerate(class_names):
    idx = y == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=name)
plt.title("PCA (2 componentes)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_2d.png", dpi=150)
plt.close()

# t-SNE (visualización no lineal)
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X))
plt.figure(figsize=(6,5))
for i, name in enumerate(class_names):
    idx = y == i
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=name)
plt.title("t-SNE (2D)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/tsne_2d.png", dpi=150)
plt.close()

# ==========================================
# 4) Partición de datos
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ==========================================
# 5) Modelos y pipelines
# ==========================================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, multi_class="auto", random_state=RANDOM_STATE))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "DecisionTree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n=== Validación cruzada (accuracy media ± std) ===")
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"{name:>15}: {scores.mean():.3f} ± {scores.std():.3f}")

# ==========================================
# 6) Ajuste de hiperparámetros (SVM y RandomForest)
# ==========================================
param_grid_svm = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.1, 0.01, 0.001]
}
svm_grid = GridSearchCV(models["SVM_RBF"], param_grid_svm, cv=cv, scoring="accuracy", n_jobs=-1)
svm_grid.fit(X_train, y_train)
print("\nMejor SVM:", svm_grid.best_params_, "accuracy=", svm_grid.best_score_)

param_grid_rf = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 3, 5, 7]
}
rf_grid = GridSearchCV(models["RandomForest"], param_grid_rf, cv=cv, scoring="accuracy", n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("Mejor RF:", rf_grid.best_params_, "accuracy=", rf_grid.best_score_)

best_estimator = svm_grid if svm_grid.best_score_ >= rf_grid.best_score_ else rf_grid
best_model = best_estimator.best_estimator_
best_name = "SVM_RBF" if best_estimator is svm_grid else "RandomForest"
print(f"\nModelo seleccionado: {best_name}")

# ==========================================
# 7) Evaluación en test
# ==========================================
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print("\n=== Resultados en test ===")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=class_names))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title(f"Matriz de confusión - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# ROC-AUC multi-clase (OvR)
try:
    y_score = best_model.predict_proba(X_test)
    y_bin = pd.get_dummies(y_test).values
    auc = roc_auc_score(y_bin, y_score, multi_class='ovr', average='macro')
    print(f"ROC-AUC (macro, OvR): {auc:.3f}")
except Exception as e:
    print("No se pudo calcular ROC-AUC:", e)

# ==========================================
# 8) Importancia de características
# ==========================================
feature_importances = None
feature_names = X_feat.columns
try:
    clf = best_model.named_steps.get('clf')
    if hasattr(clf, 'feature_importances_'):
        feature_importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        coefs = clf.coef_
        feature_importances = np.mean(np.abs(coefs), axis=0)
except Exception:
    feature_importances = None

if feature_importances is not None:
    imp = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    sns.barplot(x=imp.values, y=imp.index, palette="mako")
    plt.title(f"Importancia de características - {best_name}")
    plt.xlabel("Importancia (relativa)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importances.png", dpi=150)
    plt.close()

# ==========================================
# 9) Guardar el mejor modelo
# ==========================================
from joblib import dump
os.makedirs('outputs', exist_ok=True)
dump(best_model, "outputs/iris_best_model.joblib")
print("\nModelo guardado en outputs/iris_best_model.joblib")

with open("outputs/summary.txt", "w", encoding="utf-8") as f:
    f.write("Resultados del análisis Iris\n")
    f.write(f"Modelo: {best_name}\n")
    f.write(f"Accuracy test: {acc:.3f}\n")
    f.write(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}\n")

print("\nGráficos y artefactos guardados en la carpeta 'outputs'.")