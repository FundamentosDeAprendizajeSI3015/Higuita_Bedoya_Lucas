"""
╔══════════════════════════════════════════════════════════════════╗
║         CLUSTERING PIPELINE: Fuzzy C-Means + Subtractive        ║
║              with UMAP & t-SNE Visualizations                   ║
╚══════════════════════════════════════════════════════════════════╝

Input CSV format:
  - Shape: (n+1) x m  → n feature rows + optional header, m samples (columns)
    OR standard (m x n): m rows = samples, n columns = features  ← auto-detected
  - The script auto-detects orientation based on shape.

Usage:
  python clustering_pipeline.py --csv data.csv [options]

Options:
  --csv         Path to input CSV file                   (required)
  --n-clusters  Number of clusters for FCM               (default: 3)
  --fuzziness   Fuzziness exponent m for FCM ≥ 1         (default: 2.0)
  --ra          Subtractive: neighbourhood radius ra     (default: 0.5)
  --rb          Subtractive: inhibit radius rb           (default: 0.75)
  --eps-upper   Subtractive: accept threshold            (default: 0.5)
  --eps-lower   Subtractive: reject threshold            (default: 0.15)
  --output-dir  Directory to save plots                  (default: ./output)
  --seed        Random seed                              (default: 42)
  --header      Row index to use as header (default: None / infer)
  --transpose   Force transpose of CSV data
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── optional heavy deps ───────────────────────────────────────────────────────
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] 'umap-learn' not installed. UMAP plots will be skipped.\n"
          "       Install with: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] 'scikit-learn' not installed – several features disabled.\n"
          "       Install with: pip install scikit-learn")

# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE  (neon-on-dark, perceptually distinct)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A29BFE", "#FD79A8",
    "#00CEC9", "#FDCB6E", "#6C5CE7", "#00B894", "#E17055",
    "#74B9FF", "#55EFC4", "#FAB1A0", "#81ECEC", "#DFE6E9",
]
BG_COLOR   = "#0D1117"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#E6EDF3"
ACCENT     = "#58A6FF"

# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  GRID_COLOR,
    "legend.framealpha": 0.9,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "figure.dpi":        150,
})


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(csv_path: str, force_transpose: bool = False) -> np.ndarray:
    """
    Load CSV → numpy array of shape (m_samples, n_features).
    Auto-detects orientation: if columns >> rows, assumes data is (n x m) and transposes.
    """
    print(f"\n{'─'*60}")
    print(f"  Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, header=0)

    # Try to cast all to float; drop non-numeric cols
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] == 0:
        # Maybe the CSV has no header row
        df = pd.read_csv(csv_path, header=None)
        df_numeric = df.select_dtypes(include=[np.number])

    X = df_numeric.values.astype(float)

    # Replace infinities and impute missing values per feature.
    X = np.where(np.isfinite(X), X, np.nan)
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_rows, nan_cols = np.where(np.isnan(X))
        X[nan_rows, nan_cols] = col_medians[nan_cols]
        print(f"  [INFO] Missing values detected: {nan_count}. Imputed with column medians.")

    if force_transpose:
        X = X.T

    # Auto-orient: if features >> samples, transpose
    elif X.shape[1] > X.shape[0] * 2:
        print(f"  [INFO] Detected shape {X.shape} → transposing to ({X.shape[1]}, {X.shape[0]})")
        X = X.T

    print(f"  Data shape: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Value range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"{'─'*60}\n")
    return X


# ══════════════════════════════════════════════════════════════════════════════
#  2. SUBTRACTIVE CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
class SubtractiveClustering:
    """
    Mountain / Subtractive Clustering (Chiu, 1994).

    Parameters
    ----------
    ra : float   neighbourhood radius for potential computation
    rb : float   inhibit radius  (typically 1.25 * ra  or  1.5 * ra)
    eps_upper : float   high potential threshold → accept centre
    eps_lower : float   low  potential threshold → reject & stop
    """

    def __init__(self, ra=0.5, rb=0.75, eps_upper=0.5, eps_lower=0.15):
        self.ra        = ra
        self.rb        = rb
        self.eps_upper = eps_upper
        self.eps_lower = eps_lower
        self.centers_  = None
        self.n_clusters_ = 0

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _potential(X: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        """Compute potential contribution of a single centre on all points."""
        dist_sq = np.sum(((X - center) / (radius / 2)) ** 2, axis=1)
        return np.exp(-dist_sq)

    def fit(self, X: np.ndarray):
        """Run subtractive clustering on X (m_samples, n_features)."""
        t0 = time.time()
        print("  ┌─ Subtractive Clustering ─────────────────────────────")
        print(f"  │  ra={self.ra}  rb={self.rb}  "
              f"ε_up={self.eps_upper}  ε_lo={self.eps_lower}")

        X_norm, self._x_min, self._x_range = self._normalise(X)

        # Initial potentials
        D_a = np.zeros(len(X_norm))
        for xi in X_norm:
            D_a += self._potential(X_norm, xi, self.ra)

        centers_norm = []
        D = D_a.copy()
        D1_max = D.max()

        if not np.isfinite(D1_max) or D1_max <= 0:
            print("  │  [WARN] Invalid initial potential (<=0 or NaN). No centres found.")
            self.centers_ = np.empty((0, X.shape[1]))
            self.n_clusters_ = 0
            elapsed = time.time() - t0
            print(f"  │  Found {self.n_clusters_} centres in {elapsed:.3f}s")
            print("  └─────────────────────────────────────────────────────\n")
            return self

        iteration = 0
        while True:
            iteration += 1
            idx  = np.argmax(D)
            P_k  = D[idx]
            c_k  = X_norm[idx]

            if not np.isfinite(P_k) or P_k <= 0:
                print(f"  │  iter {iteration:3d}: potential={P_k} → no valid potential left → STOP")
                break

            ratio = P_k / D1_max
            if not np.isfinite(ratio):
                print(f"  │  iter {iteration:3d}: ratio={ratio} → invalid ratio → STOP")
                break

            if   ratio > self.eps_upper:
                accept = True
            elif ratio < self.eps_lower:
                accept = False
                print(f"  │  iter {iteration:3d}: ratio={ratio:.4f} → below ε_lower → STOP")
                break
            else:
                # Check if acceptable based on distance to boundary
                d_min = min(
                    np.linalg.norm(c_k - c) for c in centers_norm
                ) if centers_norm else np.inf
                accept = (d_min / self.ra + ratio) >= 1.0
                if not accept:
                    D[idx] = 0.0
                    print(f"  │  iter {iteration:3d}: ratio={ratio:.4f} → squashed")
                    continue

            if accept:
                centers_norm.append(c_k)
                print(f"  │  iter {iteration:3d}: ratio={ratio:.4f} → ACCEPTED centre #{len(centers_norm)}")
                # Inhibit potentials around new centre
                D -= P_k * self._potential(X_norm, c_k, self.rb)
                D = np.clip(D, 0, None)
                if D.max() == 0:
                    break
            else:
                print(f"  │  iter {iteration:3d}: ratio={ratio:.4f} → REJECTED")
                break

        # Denormalise
        self.centers_ = np.array([
            c * self._x_range + self._x_min for c in centers_norm
        ])
        self.n_clusters_ = len(self.centers_)
        elapsed = time.time() - t0
        print(f"  │  Found {self.n_clusters_} centres in {elapsed:.3f}s")
        print("  └─────────────────────────────────────────────────────\n")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its nearest centre."""
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in self.centers_])
        return np.argmin(dists, axis=0)

    @staticmethod
    def _normalise(X):
        x_min   = X.min(axis=0)
        x_range = X.max(axis=0) - x_min
        x_range[x_range == 0] = 1.0
        return (X - x_min) / x_range, x_min, x_range


# ══════════════════════════════════════════════════════════════════════════════
#  3. FUZZY C-MEANS
# ══════════════════════════════════════════════════════════════════════════════
class FuzzyCMeans:
    """
    Fuzzy C-Means clustering (Bezdek, 1981).

    Parameters
    ----------
    n_clusters : int    number of clusters c
    m          : float  fuzziness exponent (m > 1; typically 2)
    max_iter   : int    maximum iterations
    tol        : float  convergence tolerance on membership change
    init       : str    'random' | 'subtractive' – initialise centres
    sub_ra     : float  ra for subtractive init
    """

    def __init__(self, n_clusters=3, m=2.0, max_iter=300, tol=1e-6,
                 init="random", sub_ra=0.5, random_state=42):
        self.n_clusters   = n_clusters
        self.n_clusters_  = n_clusters
        self.m            = m
        self.max_iter     = max_iter
        self.tol          = tol
        self.init         = init
        self.sub_ra       = sub_ra
        self.random_state = random_state

        self.centers_    = None
        self.U_          = None   # membership matrix (n_samples, c)
        self.history_    = []     # objective function per iteration
        self.n_iter_     = 0

    # ── internals ─────────────────────────────────────────────────────────────
    def _init_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U   = rng.random((n_samples, self.n_clusters))
        U  /= U.sum(axis=1, keepdims=True)
        return U

    def _update_centers(self, X, U):
        um = U ** self.m                                   # (m, c)
        return (um.T @ X) / um.sum(axis=0)[:, None]       # (c, n_feat)

    def _update_membership(self, X, centers):
        n, c = len(X), len(centers)
        # dist[i, k] = ||x_i - v_k||
        dist = np.array([np.linalg.norm(X - centers[k], axis=1)
                         for k in range(c)]).T               # (n, c)
        dist = np.fmax(dist, np.finfo(float).eps)
        exp  = 2.0 / (self.m - 1)
        # U[i,k] = 1 / sum_j (d_ik/d_ij)^exp
        U = np.zeros((n, c))
        for k in range(c):
            ratio = dist[:, k:k+1] / dist            # (n, c)
            U[:, k] = 1.0 / (ratio ** exp).sum(axis=1)
        return U

    def _objective(self, X, U, centers):
        dist_sq = np.array([np.sum((X - centers[k]) ** 2, axis=1)
                            for k in range(len(centers))]).T
        return np.sum((U ** self.m) * dist_sq)

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, init_centers=None):
        t0 = time.time()
        print("  ┌─ Fuzzy C-Means ───────────────────────────────────────")
        print(f"  │  c={self.n_clusters}  m={self.m}  "
              f"max_iter={self.max_iter}  tol={self.tol}")

        if init_centers is not None and len(init_centers) == self.n_clusters:
            centers = init_centers.copy()
            print("  │  Init: using provided centres (from Subtractive)")
            U = self._update_membership(X, centers)
        else:
            U       = self._init_membership(len(X))
            centers = self._update_centers(X, U)
            print(f"  │  Init: {self.init}")

        for it in range(1, self.max_iter + 1):
            U_old   = U.copy()
            centers = self._update_centers(X, U)
            U       = self._update_membership(X, centers)
            J       = self._objective(X, U, centers)
            self.history_.append(J)
            delta   = np.max(np.abs(U - U_old))

            if it % 20 == 0 or it == 1:
                print(f"  │  iter {it:4d}: J={J:.6f}  ΔU={delta:.2e}")

            if delta < self.tol:
                print(f"  │  Converged at iter {it}  (ΔU={delta:.2e})")
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter
            print(f"  │  Reached max_iter={self.max_iter}")

        self.centers_ = centers
        self.U_       = U
        self.n_clusters_ = self.n_clusters
        elapsed       = time.time() - t0
        print(f"  │  Done in {elapsed:.3f}s")
        print("  └─────────────────────────────────────────────────────\n")
        return self

    def predict(self, X=None) -> np.ndarray:
        """Hard assignment: argmax over membership."""
        if X is not None:
            U = self._update_membership(X, self.centers_)
        else:
            U = self.U_
        return np.argmax(U, axis=1)

    def fit_predict(self, X, init_centers=None):
        return self.fit(X, init_centers).predict()


# ══════════════════════════════════════════════════════════════════════════════
#  4. METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(X, labels, algorithm_name):
    metrics = {}
    if not HAS_SKLEARN:
        return metrics
    unique = np.unique(labels)
    n_unique = len(unique)
    n_samples = len(X)
    if n_unique < 2 or n_unique >= n_samples:
        print(
            f"  [WARN] {algorithm_name}: invalid cluster count ({n_unique}) "
            f"for metrics with {n_samples} samples – metrics skipped."
        )
        return metrics
    metrics["silhouette"]    = silhouette_score(X, labels)
    metrics["calinski"]      = calinski_harabasz_score(X, labels)
    metrics["davies_bouldin"]= davies_bouldin_score(X, labels)
    print(f"\n  ── {algorithm_name} Metrics ─────────────────────────────")
    print(f"  Silhouette Score   : {metrics['silhouette']:.4f}  (↑ better, max=1)")
    print(f"  Calinski-Harabasz  : {metrics['calinski']:.4f}  (↑ better)")
    print(f"  Davies-Bouldin     : {metrics['davies_bouldin']:.4f}  (↓ better)")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  5. DIMENSIONALITY REDUCTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def run_tsne(X, seed=42):
    if not HAS_SKLEARN:
        return None
    perp = min(30, max(5, len(X) // 5))
    print(f"  Running t-SNE  (perplexity={perp}) …")
    reducer = TSNE(n_components=2, perplexity=perp, random_state=seed,
                   max_iter=1000, init="pca")
    return reducer.fit_transform(X)

def run_umap(X, seed=42):
    if not HAS_UMAP:
        return None
    n_neighbors = min(15, len(X) - 1)
    print(f"  Running UMAP   (n_neighbors={n_neighbors}) …")
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                   min_dist=0.1, random_state=seed)
    return reducer.fit_transform(X)


# ══════════════════════════════════════════════════════════════════════════════
#  6. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def _scatter(ax, Z, labels, n_clusters, title, subtitle="", alpha=0.75,
             membership=None, show_density=False):
    """Core scatter – colour by cluster, optional alpha-by-membership."""
    colors = [PALETTE[k % len(PALETTE)] for k in range(n_clusters)]

    for k in range(n_clusters):
        mask = labels == k
        if not mask.any():
            continue
        pts  = Z[mask]
        alph = alpha
        if membership is not None:
            # Per-point alpha proportional to max membership
            alph_arr = 0.3 + 0.7 * membership[mask, k]
            ax.scatter(pts[:, 0], pts[:, 1], c=colors[k],
                       alpha=alph_arr, s=22, linewidths=0,
                       label=f"Cluster {k+1}")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], c=colors[k],
                       alpha=alph, s=22, linewidths=0,
                       label=f"Cluster {k+1}")

    ax.set_title(title, color=TEXT_COLOR)
    if subtitle:
        ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
                ha="center", color="#8B949E", fontsize=9)
    ax.legend(loc="best", markerscale=1.5, fontsize=8,
              labelcolor=TEXT_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def _add_centres(ax, Z_centres):
    """Overlay cluster centres as ★ markers."""
    for zc in Z_centres:
        ax.scatter(*zc, marker="*", s=280, c="white",
                   edgecolors=ACCENT, linewidths=1.2, zorder=10)


def plot_objective(history, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    iters = np.arange(1, len(history) + 1)
    ax.plot(iters, history, color=ACCENT, lw=2, label="Objective J(U,V)")
    ax.fill_between(iters, history, alpha=0.15, color=ACCENT)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function J")
    ax.set_title("FCM — Objective Function Convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_membership_heatmap(U, out_path, max_points=200):
    """Plot membership matrix as heatmap (sub-sampled if large)."""
    if len(U) > max_points:
        idx = np.random.choice(len(U), max_points, replace=False)
        idx.sort()
        U_plot = U[idx]
        subtitle = f"(random sample of {max_points}/{len(U)} points)"
    else:
        U_plot  = U
        subtitle = ""

    fig, ax = plt.subplots(figsize=(max(6, U.shape[1]*0.8), 5))
    cmap = LinearSegmentedColormap.from_list(
        "fcm", ["#0D1117", ACCENT, "#FFE66D"])
    im = ax.imshow(U_plot.T, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(U.shape[1]))
    ax.set_yticklabels([f"C{k+1}" for k in range(U.shape[1])])
    ax.set_title("FCM Membership Matrix  " + subtitle)
    plt.colorbar(im, ax=ax, label="Membership degree μ")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def build_main_figure(
    Z_tsne, Z_umap,
    labels_fcm, labels_sub,
    U_fcm,
    n_clusters_fcm, n_clusters_sub,
    metrics_fcm, metrics_sub,
    out_path
):
    n_rows = 2
    n_cols = 2 if (Z_tsne is not None and Z_umap is not None) else 1
    total  = n_rows * n_cols

    # Determine active panels
    panels = []
    for method, Z in [("t-SNE", Z_tsne), ("UMAP", Z_umap)]:
        if Z is not None:
            for algo, lab, nc, U in [
                ("Fuzzy C-Means", labels_fcm, n_clusters_fcm, U_fcm),
                ("Subtractive",  labels_sub,  n_clusters_sub,  None),
            ]:
                panels.append((method, Z, algo, lab, nc, U))

    n_panels = len(panels)
    cols = 2 if n_panels > 1 else 1
    rows = (n_panels + 1) // 2

    fig = plt.figure(figsize=(7 * cols, 6.5 * rows))
    fig.patch.set_facecolor(BG_COLOR)

    # Title banner
    fig.text(0.5, 0.995, "Clustering Pipeline  |  Fuzzy C-Means & Subtractive",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color=TEXT_COLOR)
    fig.text(0.5, 0.977, "Dimensionality reduction via t-SNE / UMAP",
             ha="center", va="top", fontsize=10, color="#8B949E")

    axes = []
    for i, (method, Z, algo, lab, nc, U) in enumerate(panels):
        ax = fig.add_subplot(rows, cols, i + 1)
        axes.append(ax)

        mem = U if (U is not None and U.shape[1] == nc) else None
        subtitle = _metric_subtitle(
            metrics_fcm if algo == "Fuzzy C-Means" else metrics_sub)
        _scatter(ax, Z, lab, nc,
                 title=f"{algo}  ×  {method}",
                 subtitle=subtitle,
                 membership=mem)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _metric_subtitle(m):
    if not m:
        return ""
    parts = []
    if "silhouette" in m:
        parts.append(f"Sil={m['silhouette']:.3f}")
    if "calinski" in m:
        parts.append(f"CH={m['calinski']:.1f}")
    if "davies_bouldin" in m:
        parts.append(f"DB={m['davies_bouldin']:.3f}")
    return "  |  ".join(parts)


def plot_cluster_profiles(X, labels, n_clusters, algo_name, out_path):
    """Parallel coordinates / radar-style mean profile per cluster."""
    n_feat = X.shape[1]
    feat_labels = [f"F{i+1}" for i in range(n_feat)]

    fig, ax = plt.subplots(figsize=(max(8, n_feat * 0.7), 5))
    x_pos = np.arange(n_feat)

    for k in range(n_clusters):
        mask = labels == k
        if not mask.any():
            continue
        mean = X[mask].mean(axis=0)
        std  = X[mask].std(axis=0)
        c    = PALETTE[k % len(PALETTE)]
        ax.plot(x_pos, mean, color=c, lw=2.2, marker="o",
                markersize=5, label=f"Cluster {k+1}  (n={mask.sum()})")
        ax.fill_between(x_pos, mean - std, mean + std,
                        color=c, alpha=0.12)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(feat_labels, rotation=45 if n_feat > 10 else 0)
    ax.set_title(f"{algo_name} — Mean Feature Profiles (±1 σ)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_umap_standalone(Z, labels, n_clusters, U, title, out_path):
    """Large, publication-quality UMAP standalone figure."""
    fig, ax = plt.subplots(figsize=(9, 8))
    _scatter(ax, Z, labels, n_clusters, title=title, membership=U,
             alpha=0.80)
    ax.set_title(title, fontsize=16, pad=15)
    _watermark(fig)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_tsne_standalone(Z, labels, n_clusters, U, title, out_path):
    """Large, publication-quality t-SNE standalone figure."""
    plot_umap_standalone(Z, labels, n_clusters, U, title, out_path)


def _watermark(fig):
    fig.text(0.99, 0.01, "Clustering Pipeline  •  FCM + Subtractive",
             ha="right", va="bottom", fontsize=7, color="#484F58",
             style="italic")


# ══════════════════════════════════════════════════════════════════════════════
#  7. PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    X = load_data(args.csv, force_transpose=args.transpose)

    # ── Scale ────────────────────────────────────────────────────────────────
    if HAS_SKLEARN:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    # ── Subtractive Clustering ───────────────────────────────────────────────
    sub = SubtractiveClustering(
        ra=args.ra, rb=args.rb,
        eps_upper=args.eps_upper, eps_lower=args.eps_lower
    ).fit(X_scaled)

    if sub.n_clusters_ == 0:
        print("[WARN] Subtractive found 0 clusters. Falling back to 1 centre = global mean.")
        sub.centers_ = X_scaled.mean(axis=0, keepdims=True)
        sub.n_clusters_ = 1

    labels_sub = sub.predict(X_scaled)

    # ── Fuzzy C-Means ────────────────────────────────────────────────────────
    #  If subtractive found enough centres, use them as init; else use random
    n_fcm = args.n_clusters
    init_centres = sub.centers_ if sub.n_clusters_ == n_fcm else None

    fcm = FuzzyCMeans(
        n_clusters=n_fcm,
        m=args.fuzziness,
        max_iter=300,
        tol=1e-6,
        init="subtractive" if init_centres is not None else "random",
        random_state=args.seed
    ).fit(X_scaled, init_centers=init_centres)

    labels_fcm = fcm.predict()
    U_fcm      = fcm.U_

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics_fcm = compute_metrics(X_scaled, labels_fcm, "Fuzzy C-Means")
    metrics_sub = compute_metrics(X_scaled, labels_sub, "Subtractive")

    # ── Dimensionality Reduction ─────────────────────────────────────────────
    print("\n  ── Dimensionality Reduction ──────────────────────────────")
    Z_tsne = run_tsne(X_scaled, seed=args.seed)
    Z_umap = run_umap(X_scaled, seed=args.seed)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n  ── Generating Plots ──────────────────────────────────────")

    # 1) Main comparison grid
    build_main_figure(
        Z_tsne, Z_umap,
        labels_fcm, labels_sub,
        U_fcm,
        fcm.n_clusters_, sub.n_clusters_,
        metrics_fcm, metrics_sub,
        out_path=out_dir / "01_main_comparison.png"
    )

    # 2) FCM objective convergence
    plot_objective(fcm.history_, out_dir / "02_fcm_convergence.png")

    # 3) FCM membership heatmap
    plot_membership_heatmap(U_fcm, out_dir / "03_fcm_membership.png")

    # 4 & 5) Standalone UMAP
    if Z_umap is not None:
        plot_umap_standalone(
            Z_umap, labels_fcm, fcm.n_clusters_, U_fcm,
            title=f"UMAP  ×  Fuzzy C-Means  (c={fcm.n_clusters_})",
            out_path=out_dir / "04_umap_fcm.png"
        )
        plot_umap_standalone(
            Z_umap, labels_sub, sub.n_clusters_, None,
            title=f"UMAP  ×  Subtractive  (c={sub.n_clusters_})",
            out_path=out_dir / "05_umap_subtractive.png"
        )

    # 6 & 7) Standalone t-SNE
    if Z_tsne is not None:
        plot_tsne_standalone(
            Z_tsne, labels_fcm, fcm.n_clusters_, U_fcm,
            title=f"t-SNE  ×  Fuzzy C-Means  (c={fcm.n_clusters_})",
            out_path=out_dir / "06_tsne_fcm.png"
        )
        plot_tsne_standalone(
            Z_tsne, labels_sub, sub.n_clusters_, None,
            title=f"t-SNE  ×  Subtractive  (c={sub.n_clusters_})",
            out_path=out_dir / "07_tsne_subtractive.png"
        )

    # 8 & 9) Feature profiles
    plot_cluster_profiles(X, labels_fcm, fcm.n_clusters_,
                          "Fuzzy C-Means",
                          out_dir / "08_fcm_profiles.png")
    plot_cluster_profiles(X, labels_sub, sub.n_clusters_,
                          "Subtractive",
                          out_dir / "09_sub_profiles.png")

    # ── Save labels CSV ──────────────────────────────────────────────────────
    result_df = pd.DataFrame({
        "sample_idx":       np.arange(len(X)),
        "cluster_fcm":      labels_fcm,
        "cluster_sub":      labels_sub,
        "fcm_max_membership": U_fcm.max(axis=1).round(4)
    })
    for k in range(fcm.n_clusters_):
        result_df[f"fcm_mu_{k+1}"] = U_fcm[:, k].round(4)

    labels_csv = out_dir / "cluster_assignments.csv"
    result_df.to_csv(labels_csv, index=False)
    print(f"\n  Saved cluster assignments → {labels_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'═'*60}")
    print(f"  Input          : {args.csv}")
    print(f"  Samples        : {len(X)}")
    print(f"  Features       : {X.shape[1]}")
    print(f"  ──────────────────────────────────────────────────────")
    print(f"  Subtractive    : {sub.n_clusters_} clusters  "
          f"(ra={args.ra}, rb={args.rb})")
    print(f"  Fuzzy C-Means  : {fcm.n_clusters_} clusters  "
          f"(m={args.fuzziness}, iters={fcm.n_iter_})")
    if metrics_fcm:
        print(f"  FCM  Sil       : {metrics_fcm.get('silhouette', 'N/A'):.4f}")
    if metrics_sub:
        print(f"  Sub  Sil       : {metrics_sub.get('silhouette', 'N/A'):.4f}")
    print(f"  Output dir     : {out_dir.resolve()}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  8. CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Clustering pipeline: Fuzzy C-Means + Subtractive, "
                    "with UMAP and t-SNE visualisations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--csv",        required=True,  help="Path to input CSV")
    p.add_argument("--n-clusters", type=int,   default=3,    dest="n_clusters")
    p.add_argument("--fuzziness",  type=float, default=2.0)
    p.add_argument("--ra",         type=float, default=0.5)
    p.add_argument("--rb",         type=float, default=0.75)
    p.add_argument("--eps-upper",  type=float, default=0.5,  dest="eps_upper")
    p.add_argument("--eps-lower",  type=float, default=0.15, dest="eps_lower")
    p.add_argument("--output-dir", type=str,   default="./output", dest="output_dir")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--transpose",  action="store_true",
                   help="Force transpose of CSV (use if features are rows)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
