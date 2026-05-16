from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BINARY_TARGET_COL,
    MODELING_EXCLUDE_COLS,
    REPORT_IMAGES_DIR,
    SEED,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    ensure_directories,
)
from src.utils.logging_setup import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def load_xy_from_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c not in MODELING_EXCLUDE_COLS]
    x = df[cols].to_numpy(dtype=np.float64)
    y = df[BINARY_TARGET_COL].to_numpy(dtype=int)
    return x, y


def plot_pca_auc_curve(
    train_path: Path = TRAIN_PARQUET_PATH,
    val_path: Path = VAL_PARQUET_PATH,
    components: list[int] | None = None,
    out_path: Path = REPORT_IMAGES_DIR / "06_pca_auc_curve.png",
) -> pd.DataFrame:
    set_global_seed(SEED)
    ensure_directories()
    x_tr, y_tr = load_xy_from_parquet(train_path)
    x_va, y_va = load_xy_from_parquet(val_path)
    max_c = min(x_tr.shape[1], x_tr.shape[0] - 1)
    if components is None:
        components = sorted({5, 10, 20, 30, min(40, max_c), max_c})
    components = [c for c in components if c <= max_c]

    rows: list[dict[str, float]] = []
    for k in components:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=k, random_state=SEED)),
                ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]
        )
        pipe.fit(x_tr, y_tr)
        proba = pipe.predict_proba(x_va)[:, 1]
        auc = float(roc_auc_score(y_va, proba))
        pca = pipe.named_steps["pca"]
        ev = float(np.sum(pca.explained_variance_ratio_))
        rows.append({"n_components": float(k), "roc_auc_val": auc, "explained_variance": ev})
        log.info("PCA k=%d | val ROC-AUC=%.4f | explained_var=%.3f", k, auc, ev)

    tab = pd.DataFrame(rows)
    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(tab["n_components"], tab["roc_auc_val"], "o-", color="tab:blue")
    ax1.set_xlabel("n_components (PCA)")
    ax1.set_ylabel("ROC-AUC (val)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(tab["n_components"], tab["explained_variance"], "s--", color="tab:orange", alpha=0.7)
    ax2.set_ylabel("Explained variance ratio (sum)", color="tab:orange")
    fig.suptitle("LogReg на PCA-признаках: качество vs размерность")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Сохранён график %s", out_path)
    return tab


def plot_umap_scatter(
    train_path: Path = TRAIN_PARQUET_PATH,
    sample_size: int = 4000,
    out_path: Path = REPORT_IMAGES_DIR / "06_umap_scatter.png",
) -> None:
    set_global_seed(SEED)
    try:
        import umap
    except ImportError:
        log.warning("umap-learn не установлен — пропуск UMAP-графика")
        return

    x_tr, y_tr = load_xy_from_parquet(train_path)
    n = min(sample_size, len(x_tr))
    idx = np.random.choice(len(x_tr), size=n, replace=False)
    x_s = x_tr[idx]
    y_s = y_tr[idx]

    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
    emb = reducer.fit_transform(x_s)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y_s, cmap="coolwarm", alpha=0.35, s=8)
    ax.set_title("UMAP-2D (подвыборка train, окраска=is_popular)")
    fig.colorbar(scatter, ax=ax, label="is_popular")
    fig.tight_layout()
    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("UMAP график: %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Визуализации снижения размерности.")
    p.add_argument("--train", type=Path, default=TRAIN_PARQUET_PATH)
    p.add_argument("--val", type=Path, default=VAL_PARQUET_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot_pca_auc_curve(train_path=args.train, val_path=args.val)
    plot_umap_scatter(train_path=args.train)


if __name__ == "__main__":
    main()
