import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MAX_RELABEL_RATE = 0.30


def save_fig(output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
    plt.close()


def generate_plots(
    df,
    df_results,
    num_cols_cluster,
    clustering_outputs,
    metrics_df,
    classification_df,
    regression_df,
    output_dir,
):
    sns.set_theme(style="whitegrid")

    # 1) Distribucion de categorias originales.
    plt.figure(figsize=(10, 5))
    order_orig = df["categoria"].value_counts().index
    sns.countplot(
        data=df,
        y="categoria",
        order=order_orig,
        hue="categoria",
        palette="viridis",
        legend=False,
    )
    plt.title("Distribucion de categorias originales")
    plt.xlabel("Cantidad")
    plt.ylabel("Categoria")
    save_fig(output_dir, "01_distribucion_categorias_original.png")

    # 2) Distribucion de categorias reevaluadas.
    plt.figure(figsize=(10, 5))
    order_re = df_results["categoria_reevaluada"].value_counts().index
    sns.countplot(
        data=df_results,
        y="categoria_reevaluada",
        order=order_re,
        hue="categoria_reevaluada",
        palette="magma",
        legend=False,
    )
    plt.title("Distribucion de categorias reevaluadas")
    plt.xlabel("Cantidad")
    plt.ylabel("Categoria")
    save_fig(output_dir, "02_distribucion_categorias_reevaluada.png")

    # 3) Cantidad de cambios por categoria original.
    changed = df_results[df_results["categoria"] != df_results["categoria_reevaluada"]]
    plt.figure(figsize=(10, 5))
    if not changed.empty:
        sns.countplot(
            data=changed,
            y="categoria",
            hue="categoria",
            palette="rocket",
            legend=False,
        )
    plt.title("Registros con etiqueta cambiada por categoria original")
    plt.xlabel("Cantidad de cambios")
    plt.ylabel("Categoria")
    save_fig(output_dir, "03_cambios_por_categoria.png")

    # 4) Histograma de confianza de consenso.
    plt.figure(figsize=(9, 5))
    sns.histplot(df_results["confianza_consenso"], bins=20, kde=True, color="#2a9d8f")
    plt.axvline(0.60, color="red", linestyle="--", label="Umbral 0.60")
    plt.title("Distribucion de confianza del consenso")
    plt.xlabel("Confianza")
    plt.ylabel("Frecuencia")
    plt.legend()
    save_fig(output_dir, "04_histograma_confianza_consenso.png")

    # 5) Matriz de correlacion de variables numericas.
    corr = df[num_cols_cluster].corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Mapa de calor de correlaciones")
    save_fig(output_dir, "05_heatmap_correlaciones.png")

    # 6) Comparacion de metricas de clustering.
    plot_df = metrics_df.copy()
    for col in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
        col_max = plot_df[col].max(skipna=True)
        col_min = plot_df[col].min(skipna=True)
        if pd.notna(col_max) and col_max != col_min:
            plot_df[f"{col}_norm"] = (plot_df[col] - col_min) / (col_max - col_min)
        else:
            plot_df[f"{col}_norm"] = np.nan
    plot_df["davies_bouldin_inv_norm"] = 1 - plot_df["davies_bouldin_norm"]
    melt_cols = ["silhouette_norm", "calinski_harabasz_norm", "davies_bouldin_inv_norm"]
    metrics_melt = plot_df.melt(id_vars="metodo", value_vars=melt_cols)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=metrics_melt, x="metodo", y="value", hue="variable")
    plt.title("Comparacion normalizada de metricas de clustering")
    plt.xlabel("Metodo")
    plt.ylabel("Score normalizado")
    plt.xticks(rotation=20)
    save_fig(output_dir, "06_metricas_clustering_normalizadas.png")

    # 7-11) PCA para cada metodo de clustering.
    X_scaled = StandardScaler().fit_transform(df[num_cols_cluster])
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": X_2d[:, 0], "PC2": X_2d[:, 1]})

    ordered_methods = [
        "kmeans",
        "fuzzy_c_means",
        "subtractive",
        "dbscan",
        "familia_cluster_jerarquico",
    ]
    for idx, method in enumerate(ordered_methods, start=7):
        labels = clustering_outputs[method]
        plot_data = pca_df.copy()
        plot_data["cluster"] = labels.astype(str)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_data.sample(min(2000, len(plot_data)), random_state=RANDOM_STATE),
            x="PC1",
            y="PC2",
            hue="cluster",
            palette="tab20",
            s=20,
            alpha=0.7,
            linewidth=0,
        )
        plt.title(f"PCA 2D - Clusters por {method}")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        save_fig(output_dir, f"{idx:02d}_pca_clusters_{method}.png")

    # 12) Boxplot de ventas por categoria reevaluada.
    plt.figure(figsize=(12, 6))
    top_cats = df_results["categoria_reevaluada"].value_counts().head(8).index
    subset = df_results[df_results["categoria_reevaluada"].isin(top_cats)].copy()
    sns.boxplot(
        data=subset,
        x="categoria_reevaluada",
        y="ventas_ultimos_30_dias",
        hue="categoria_reevaluada",
        palette="Set2",
        legend=False,
    )
    plt.title("Ventas ultimos 30 dias por categoria reevaluada (top 8)")
    plt.xlabel("Categoria")
    plt.ylabel("Ventas ultimos 30 dias")
    plt.xticks(rotation=30, ha="right")
    save_fig(output_dir, "12_boxplot_ventas_por_categoria_reevaluada.png")

    # 13) Comparacion de accuracy de clasificacion.
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=classification_df,
        x="modelo",
        y="accuracy_vs_proxy",
        hue="dataset_entrenamiento",
        palette="Dark2",
    )
    plt.title("Accuracy vs etiqueta reevaluada (proxy)")
    plt.xlabel("Modelo")
    plt.ylabel("Accuracy")
    save_fig(output_dir, "13_accuracy_clasificacion.png")

    # 14) Comparacion de F1 macro de clasificacion.
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=classification_df,
        x="modelo",
        y="f1_macro_vs_proxy",
        hue="dataset_entrenamiento",
        palette="Set1",
    )
    plt.title("F1 Macro vs etiqueta reevaluada (proxy)")
    plt.xlabel("Modelo")
    plt.ylabel("F1 Macro")
    save_fig(output_dir, "14_f1_clasificacion.png")

    # 15) Comparacion de metricas de regresion.
    reg_plot = regression_df.melt(
        id_vars=["modelo", "dataset_entrenamiento"],
        value_vars=["mae", "rmse", "r2"],
        var_name="metrica",
        value_name="valor",
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=reg_plot,
        x="metrica",
        y="valor",
        hue="dataset_entrenamiento",
        palette="Paired",
    )
    plt.title("Comparacion de metricas de regresion lineal")
    plt.xlabel("Metrica")
    plt.ylabel("Valor")
    save_fig(output_dir, "15_metricas_regresion_lineal.png")

    # 16) Scatter precio vs demanda coloreado por categoria reevaluada.
    plt.figure(figsize=(10, 6))
    sample = df_results.sample(min(1500, len(df_results)), random_state=RANDOM_STATE)
    sns.scatterplot(
        data=sample,
        x="precio_unitario",
        y="demanda_promedio_diaria",
        hue="categoria_reevaluada",
        alpha=0.7,
        s=28,
        linewidth=0,
    )
    plt.title("Relacion precio unitario vs demanda promedio diaria")
    plt.xlabel("Precio unitario")
    plt.ylabel("Demanda promedio diaria")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_fig(output_dir, "16_scatter_precio_vs_demanda.png")

    return sorted([p.name for p in output_dir.glob("*.png")])


def fuzzy_c_means(X, c=10, m=2.0, max_iter=200, tol=1e-5, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    U = rng.random((c, n_samples))
    U = U / U.sum(axis=0, keepdims=True)

    for _ in range(max_iter):
        U_prev = U.copy()
        um = U ** m
        centers = (um @ X) / np.clip(um.sum(axis=1, keepdims=True), 1e-12, None)

        dist = np.linalg.norm(X[None, :, :] - centers[:, None, :], axis=2)
        dist = np.clip(dist, 1e-8, None)

        inv_dist = dist ** (-2 / (m - 1))
        U = inv_dist / inv_dist.sum(axis=0, keepdims=True)

        if np.max(np.abs(U - U_prev)) < tol:
            break

    labels = np.argmax(U, axis=0)
    return labels, centers, U


def subtractive_clustering(X, radius=1.2, accept_ratio=0.5, reject_ratio=0.15):
    # Basic subtractive clustering over standardized continuous space.
    n = X.shape[0]
    if n == 0:
        return np.array([]), np.empty((0, X.shape[1]))

    ra2 = (radius / 2.0) ** 2
    rb2 = (1.5 * radius / 2.0) ** 2

    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    potentials = np.sum(np.exp(-sq_dists / ra2), axis=1)

    centers_idx = []
    first_potential = potentials.max()

    while True:
        idx = int(np.argmax(potentials))
        p = potentials[idx]
        if p < reject_ratio * first_potential:
            break

        if p >= accept_ratio * first_potential:
            centers_idx.append(idx)
            d2 = np.sum((X - X[idx]) ** 2, axis=1)
            potentials = potentials - p * np.exp(-d2 / rb2)
        else:
            centers_idx.append(idx)
            d2 = np.sum((X - X[idx]) ** 2, axis=1)
            potentials = potentials - p * np.exp(-d2 / rb2)

        potentials[idx] = 0
        if len(centers_idx) >= 25:
            break

    if not centers_idx:
        centers_idx = [int(np.argmax(np.sum(X**2, axis=1)))]

    centers = X[centers_idx]
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, centers


def to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix


def cluster_metrics(X, labels):
    unique = np.unique(labels)
    valid_clusters = [c for c in unique if c != -1]
    if len(valid_clusters) < 2:
        return {
            "n_clusters": len(valid_clusters),
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "noise_rate": float(np.mean(labels == -1)) if -1 in unique else 0.0,
        }

    mask = labels != -1
    X_eval = X[mask]
    labels_eval = labels[mask]

    return {
        "n_clusters": len(np.unique(labels_eval)),
        "silhouette": float(silhouette_score(X_eval, labels_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(X_eval, labels_eval)),
        "davies_bouldin": float(davies_bouldin_score(X_eval, labels_eval)),
        "noise_rate": float(np.mean(labels == -1)) if -1 in unique else 0.0,
    }


def map_clusters_to_labels(cluster_labels, y_true):
    mapped = []
    mapping = {}
    df = pd.DataFrame({"cluster": cluster_labels, "y": y_true})

    for cl in sorted(df["cluster"].unique()):
        if cl == -1:
            continue
        mode = df[df["cluster"] == cl]["y"].mode()
        if not mode.empty:
            mapping[cl] = mode.iloc[0]

    for cl in cluster_labels:
        if cl == -1 or cl not in mapping:
            mapped.append(np.nan)
        else:
            mapped.append(mapping[cl])

    return np.array(mapped, dtype=object), mapping


def majority_vote(values):
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return np.nan, 0.0
    counter = Counter(vals)
    label, count = counter.most_common(1)[0]
    confidence = count / len(vals)
    return label, confidence


def build_supervised_features(df, category_col="categoria"):
    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "id_registro",
            "codigo_producto",
            category_col,
        }
    ]

    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return X, feature_cols, preprocessor


def main():
    base_path = Path(__file__).resolve().parent
    dataset_path = base_path / "dataset_inventario_farmacia_5000_registros.csv"

    df = pd.read_csv(dataset_path)

    y_original = df["categoria"].astype(str)

    num_cols_cluster = [
        "precio_unitario",
        "costo_unitario",
        "margen_unitario",
        "stock_actual",
        "stock_minimo",
        "stock_maximo",
        "demanda_promedio_diaria",
        "desviacion_demanda",
        "ventas_ultimos_30_dias",
        "quiebres_stock_ultimos_6m",
        "lead_time_dias",
        "dias_para_vencer",
    ]

    X_cluster = StandardScaler().fit_transform(df[num_cols_cluster])

    n_classes = y_original.nunique()

    kmeans = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=20)
    labels_kmeans = kmeans.fit_predict(X_cluster)

    labels_fcm, fcm_centers, _ = fuzzy_c_means(X_cluster, c=n_classes)

    labels_subtractive, sub_centers = subtractive_clustering(X_cluster, radius=1.1)

    dbscan = DBSCAN(eps=1.4, min_samples=10)
    labels_dbscan = dbscan.fit_predict(X_cluster)

    agg = AgglomerativeClustering(n_clusters=n_classes, linkage="ward")
    labels_agg = agg.fit_predict(X_cluster)

    clustering_outputs = {
        "kmeans": labels_kmeans,
        "fuzzy_c_means": labels_fcm,
        "subtractive": labels_subtractive,
        "dbscan": labels_dbscan,
        "familia_cluster_jerarquico": labels_agg,
    }

    metrics_rows = []
    label_predictions = {}
    label_mappings = {}

    for method, labels in clustering_outputs.items():
        m = cluster_metrics(X_cluster, labels)
        m["metodo"] = method
        metrics_rows.append(m)

        pred_label, mapping = map_clusters_to_labels(labels, y_original)
        label_predictions[method] = pred_label
        label_mappings[method] = mapping

    metrics_df = pd.DataFrame(metrics_rows)[
        [
            "metodo",
            "n_clusters",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "noise_rate",
        ]
    ]

    consensus_labels = []
    confidence_scores = []

    for i in range(len(df)):
        preds = [label_predictions[m][i] for m in clustering_outputs]
        label, conf = majority_vote(preds)
        consensus_labels.append(label)
        confidence_scores.append(conf)

    df_results = df.copy()
    df_results["categoria_consenso_cluster"] = consensus_labels
    df_results["confianza_consenso"] = confidence_scores

    mismatch_mask = (
        (df_results["categoria_consenso_cluster"].notna())
        & (df_results["categoria_consenso_cluster"] != df_results["categoria"])
        & (df_results["confianza_consenso"] >= 0.6)
    )

    mismatch_idx = df_results[mismatch_mask].sort_values(
        by="confianza_consenso", ascending=False
    ).index.tolist()

    max_changes = int(len(df_results) * MAX_RELABEL_RATE)
    selected_idx = mismatch_idx[:max_changes]

    df_results["categoria_reevaluada"] = df_results["categoria"]
    df_results.loc[selected_idx, "categoria_reevaluada"] = df_results.loc[
        selected_idx, "categoria_consenso_cluster"
    ]

    relabel_rate = (df_results["categoria_reevaluada"] != df_results["categoria"]).mean()

    # Supervised classification: compare training with original vs reevaluated labels.
    X_sup, _, preprocessor = build_supervised_features(df_results, category_col="categoria")

    y_proxy = df_results["categoria_reevaluada"].astype(str)

    X_train, X_test, y_orig_train, y_orig_test, y_proxy_train, y_proxy_test = train_test_split(
        X_sup,
        y_original,
        y_proxy,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_proxy,
    )

    clf_models = {
        "Arbol_Decision": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "Regresion_Logistica": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
        ),
    }

    classification_summary = []
    detailed_reports = {}

    for model_name, model in clf_models.items():
        pipe_orig = Pipeline([("prep", preprocessor), ("model", model)])
        pipe_orig.fit(X_train, y_orig_train)
        pred_orig = pipe_orig.predict(X_test)

        pipe_re = Pipeline([("prep", preprocessor), ("model", model)])
        pipe_re.fit(X_train, y_proxy_train)
        pred_re = pipe_re.predict(X_test)

        classification_summary.append(
            {
                "modelo": model_name,
                "dataset_entrenamiento": "original",
                "accuracy_vs_proxy": accuracy_score(y_proxy_test, pred_orig),
                "f1_macro_vs_proxy": f1_score(y_proxy_test, pred_orig, average="macro"),
                "accuracy_vs_original": accuracy_score(y_orig_test, pred_orig),
                "f1_macro_vs_original": f1_score(y_orig_test, pred_orig, average="macro"),
            }
        )

        classification_summary.append(
            {
                "modelo": model_name,
                "dataset_entrenamiento": "reevaluado",
                "accuracy_vs_proxy": accuracy_score(y_proxy_test, pred_re),
                "f1_macro_vs_proxy": f1_score(y_proxy_test, pred_re, average="macro"),
                "accuracy_vs_original": accuracy_score(y_orig_test, pred_re),
                "f1_macro_vs_original": f1_score(y_orig_test, pred_re, average="macro"),
            }
        )

        detailed_reports[f"{model_name}_original"] = classification_report(
            y_proxy_test, pred_orig, output_dict=True
        )
        detailed_reports[f"{model_name}_reevaluado"] = classification_report(
            y_proxy_test, pred_re, output_dict=True
        )

    classification_df = pd.DataFrame(classification_summary)

    # Linear regression: predict sales with original category vs reevaluated category.
    y_reg = df_results["ventas_ultimos_30_dias"]

    df_reg_orig = df_results.copy()
    df_reg_orig["categoria_modelo"] = df_reg_orig["categoria"]

    df_reg_re = df_results.copy()
    df_reg_re["categoria_modelo"] = df_reg_re["categoria_reevaluada"]

    def get_regression_xy(df_reg):
        feature_cols = [
            c
            for c in df_reg.columns
            if c
            not in {
                "id_registro",
                "codigo_producto",
                "categoria",
                "categoria_reevaluada",
                "categoria_consenso_cluster",
                "confianza_consenso",
                "ventas_ultimos_30_dias",
            }
        ]
        X = df_reg[feature_cols]
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        prep = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )
        return X, prep

    X_reg_orig, prep_reg_orig = get_regression_xy(df_reg_orig)
    X_reg_re, prep_reg_re = get_regression_xy(df_reg_re)

    idx_train, idx_test = train_test_split(
        np.arange(len(df_results)), test_size=0.2, random_state=RANDOM_STATE
    )

    reg_results = []

    for variant, X_reg, prep in [
        ("original", X_reg_orig, prep_reg_orig),
        ("reevaluado", X_reg_re, prep_reg_re),
    ]:
        pipe = Pipeline([("prep", prep), ("model", LinearRegression())])
        pipe.fit(X_reg.iloc[idx_train], y_reg.iloc[idx_train])
        pred = pipe.predict(X_reg.iloc[idx_test])
        y_true = y_reg.iloc[idx_test]

        reg_results.append(
            {
                "modelo": "Regresion_Lineal",
                "dataset_entrenamiento": variant,
                "mae": mean_absolute_error(y_true, pred),
                "rmse": root_mean_squared_error(y_true, pred),
                "r2": r2_score(y_true, pred),
            }
        )

    regression_df = pd.DataFrame(reg_results)

    plots_dir = base_path / "graficas_informe"
    generated_plots = generate_plots(
        df=df,
        df_results=df_results,
        num_cols_cluster=num_cols_cluster,
        clustering_outputs=clustering_outputs,
        metrics_df=metrics_df,
        classification_df=classification_df,
        regression_df=regression_df,
        output_dir=plots_dir,
    )

    # Save artifacts.
    metrics_df.to_csv(base_path / "resumen_clustering.csv", index=False)
    classification_df.to_csv(base_path / "resumen_modelos_clasificacion.csv", index=False)
    regression_df.to_csv(base_path / "resumen_modelos_regresion.csv", index=False)
    df_results.to_csv(base_path / "dataset_con_etiquetas_reevaluadas.csv", index=False)

    with open(base_path / "detalle_reportes_clasificacion.json", "w", encoding="utf-8") as f:
        json.dump(detailed_reports, f, ensure_ascii=False, indent=2)

    best_cluster_method = metrics_df.sort_values(
        by=["silhouette", "calinski_harabasz"], ascending=[False, False]
    ).iloc[0]["metodo"]

    md_lines = []
    md_lines.append("# Informe: Evaluacion No Supervisada y Supervisada del Inventario Farmaceutico")
    md_lines.append("")
    md_lines.append("## 1. Problema")
    md_lines.append(
        "Se requiere evaluar y mejorar la calidad de etiquetas de categoria en un inventario farmaceutico, considerando que hasta el 30% de las etiquetas puede estar mal asignado."
    )
    md_lines.append(
        "Luego, se deben entrenar modelos supervisados (Arboles de Decision, Regresion Logistica y Regresion Lineal) y compararlos entre el dataset original y el dataset con etiquetas reevaluadas."
    )
    md_lines.append("")
    md_lines.append("## 2. Dataset y Preparacion")
    md_lines.append(f"- Registros analizados: {len(df_results)}")
    md_lines.append(f"- Variables numericas para clustering: {', '.join(num_cols_cluster)}")
    md_lines.append("- Estandarizacion aplicada en variables numericas para analisis de distancias.")
    md_lines.append("")
    md_lines.append("## 3. Analisis No Supervisado")
    md_lines.append(
        "Modelos aplicados: K-Means, Fuzzy C-Means, Subtractive Clustering, DBSCAN y clustering jerarquico (familia cluster)."
    )
    md_lines.append("")
    md_lines.append("### 3.1 Metricas de Clustering")
    md_lines.append(metrics_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append(f"- Metodo con mejor separacion global (heuristica): {best_cluster_method}.")
    md_lines.append("")
    md_lines.append("## 4. Reevaluacion de Etiquetas")
    md_lines.append(
        "Se realizo voto mayoritario entre las etiquetas inferidas por los cinco metodos de clustering."
    )
    md_lines.append(
        "Solo se cambiaron etiquetas cuando hubo desacuerdo con la etiqueta original y confianza de consenso >= 0.60."
    )
    md_lines.append(
        f"Adicionalmente, se limito el cambio a maximo 30% de los registros. Tasa final de cambio: {relabel_rate:.2%}."
    )
    md_lines.append("")
    md_lines.append("## 5. Modelos Supervisados de Clasificacion")
    md_lines.append(
        "Comparacion de Arbol de Decision y Regresion Logistica entrenados con etiquetas originales vs reevaluadas."
    )
    md_lines.append("")
    md_lines.append(classification_df.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append(
        "Interpretacion clave: si el rendimiento vs proxy (etiqueta reevaluada) sube al entrenar con etiquetas reevaluadas, se evidencia mejora de consistencia en clases."
    )
    md_lines.append("")
    md_lines.append("## 6. Modelo Supervisado de Regresion")
    md_lines.append(
        "Se entreno Regresion Lineal para predecir ventas_ultimos_30_dias comparando dataset original vs reevaluado (modificando la variable de categoria como feature)."
    )
    md_lines.append("")
    md_lines.append(regression_df.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("## 7. Conclusion")
    md_lines.append(
        "El flujo no supervisado permitio detectar posibles inconsistencias de etiquetado y generar una version reevaluada del dataset."
    )
    md_lines.append(
        "La comparacion supervisada permite decidir si la reevaluacion aporta mejoras de generalizacion frente al dataset original."
    )
    md_lines.append(
        "Se recomienda usar la version reevaluada cuando mejore de forma consistente accuracy/F1 (clasificacion) y/o MAE-RMSE-R2 (regresion)."
    )
    md_lines.append("")
    md_lines.append("## 8. Graficas Generadas")
    md_lines.append("Se generaron las siguientes visualizaciones en la carpeta graficas_informe:")
    for plot_name in generated_plots:
        md_lines.append(f"- graficas_informe/{plot_name}")

    report_path = base_path / "informe_analisis_no_supervisado_supervisado.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("Analisis completado.")
    print(f"Tasa de etiquetas reevaluadas: {relabel_rate:.2%}")
    print(f"Informe generado en: {report_path}")
    print(f"Graficas generadas: {len(generated_plots)}")


if __name__ == "__main__":
    main()
