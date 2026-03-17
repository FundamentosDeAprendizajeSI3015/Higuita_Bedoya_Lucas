from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


random_state = 42
plt.rc("font", family="serif", size=12)


def detectar_codo(k_values, inertias):
    x = np.array(k_values, dtype=float)
    y = np.array(inertias, dtype=float)

    a = y[-1] - y[0]
    b = x[0] - x[-1]
    c = x[-1] * y[0] - x[0] * y[-1]

    denominador = np.sqrt(a**2 + b**2)
    if denominador == 0:
        return int(x[0])

    distancias = np.abs(a * x + b * y + c) / denominador
    indice_codo = int(np.argmax(distancias))
    return int(x[indice_codo])


def guardar_scatter(data_2d, ruta_salida, titulo, labels=None):
    fig, ax = plt.subplots()

    if labels is None:
        ax.scatter(data_2d[:, 0], data_2d[:, 1], s=18, alpha=0.85)
    else:
        ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, s=18, alpha=0.85, cmap="tab10")

    ax.set_title(titulo)
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    fig.set_size_inches(5 * 1.6, 5)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset_sintetico_FIRE_UdeA_realista.csv"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {dataset_path}")

    df = pd.read_csv(dataset_path)

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_excluir = {"label"}
    columnas_modelo = [col for col in columnas_numericas if col not in columnas_excluir]

    if len(columnas_modelo) < 2:
        raise ValueError("Se requieren al menos 2 columnas numericas para ejecutar el clustering.")

    data = df[columnas_modelo].to_numpy(dtype=float)
    print(f"Columnas usadas para clustering ({len(columnas_modelo)}): {columnas_modelo}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, np.arange(data.shape[1])),
        ]
    )

    data_scaled = preprocessor.fit_transform(data)
    data_2d = PCA(n_components=2, random_state=random_state).fit_transform(data_scaled)

    guardar_scatter(
        data_2d,
        output_dir / "01_dataset_proyectado_pca.png",
        "Dataset FIRE UdeA (proyeccion PCA en 2D)",
    )

    clu_kmeans_k2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", KMeans(n_clusters=2, random_state=random_state, n_init=10)),
        ]
    )
    clu_kmeans_k2.fit(data)
    labels_k2 = clu_kmeans_k2["clustering"].labels_
    inercia_k2 = clu_kmeans_k2["clustering"].inertia_
    print(f"Con K = 2: la inercia es {inercia_k2:.4f}")

    guardar_scatter(
        data_2d,
        output_dir / "02_kmeans_k2.png",
        "KMeans con K = 2",
        labels=labels_k2,
    )

    inert = []
    k_range = list(range(1, 11))
    for k in k_range:
        clu_kmeans = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clustering", KMeans(n_clusters=k, random_state=random_state, n_init=10)),
            ]
        )
        clu_kmeans.fit(data)
        inert.append(clu_kmeans["clustering"].inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_range, inert, marker="o")
    ax.set_title("Metodo del codo")
    ax.set_xlabel("Numero de clusters (K)")
    ax.set_ylabel("Inercia")
    fig.set_size_inches(5 * 1.6, 5)
    fig.savefig(output_dir / "03_metodo_codo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    k_optimo = detectar_codo(k_range, inert)
    print(f"Segun el metodo del codo, un K sugerido es: {k_optimo}")

    clu_kmeans_opt = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", KMeans(n_clusters=k_optimo, random_state=random_state, n_init=10)),
        ]
    )
    clu_kmeans_opt.fit(data)
    labels_kopt = clu_kmeans_opt["clustering"].labels_
    inercia_kopt = clu_kmeans_opt["clustering"].inertia_
    print(f"Con K = {k_optimo}: la inercia es {inercia_kopt:.4f}")

    guardar_scatter(
        data_2d,
        output_dir / f"04_kmeans_k{k_optimo}.png",
        f"KMeans con K = {k_optimo}",
        labels=labels_kopt,
    )

    clu_dbscan = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", DBSCAN(eps=0.1, min_samples=10)),
        ]
    )
    clu_dbscan.fit(data)
    labels_dbscan = clu_dbscan["clustering"].labels_

    guardar_scatter(
        data_2d,
        output_dir / "05_dbscan.png",
        "DBSCAN (eps=0.1, min_samples=10)",
        labels=labels_dbscan,
    )

    valores, conteos = np.unique(labels_dbscan, return_counts=True)
    distribucion = {int(v): int(c) for v, c in zip(valores, conteos)}
    print(f"Distribucion de etiquetas DBSCAN: {distribucion}")

    resultado = df.copy()
    resultado["cluster_kmeans_k2"] = labels_k2
    resultado[f"cluster_kmeans_k{k_optimo}"] = labels_kopt
    resultado["cluster_dbscan"] = labels_dbscan
    resultado.to_csv(output_dir / "clusters_resultado.csv", index=False)

    print("Proceso finalizado. Revisa la carpeta 'output' para ver graficas y resultados.")


if __name__ == "__main__":
    main()
