from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BINARY_TARGET_COL,
    MODELING_EXCLUDE_COLS,
    REPORT_TABLES_DIR,
    SEED,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    ensure_directories,
)
from src.modeling.metrics import compute_metrics
from src.modeling.tuners import (
    make_calibrated_lgbm,
    tune_knn,
    tune_lightgbm_optuna,
    tune_logistic_regression,
    tune_random_forest,
    tune_xgboost_optuna,
)
from src.utils.logging_setup import get_logger
from src.utils.mlflow_setup import setup_mlflow
from src.utils.seed import set_global_seed

log = get_logger(__name__)


@contextlib.contextmanager
def _mlflow_run(use_mlflow: bool, run_name: str) -> Iterator[None]:
    if not use_mlflow:
        yield
        return
    import mlflow

    with mlflow.start_run(run_name=run_name, nested=True):
        yield


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in MODELING_EXCLUDE_COLS]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[BINARY_TARGET_COL].to_numpy(dtype=int)
    return x, y, feature_cols


def evaluate_split(model: Any, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    proba = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)
    pred = (proba >= 0.5).astype(int)
    return compute_metrics(y, pred, proba)


def _append_row(
    rows: list[dict[str, Any]],
    *,
    name: str,
    hypothesis: str,
    model_desc: str,
    params: dict[str, Any],
    cv_auc: float | None,
    m_train: dict[str, float],
    m_val: dict[str, float],
    m_test: dict[str, float],
) -> None:
    rows.append(
        {
            "experiment": name,
            "hypothesis": hypothesis,
            "model": model_desc,
            "cv_roc_auc": cv_auc,
            "params_json": json.dumps(params, ensure_ascii=False, default=str),
            **{f"train_{k}": v for k, v in m_train.items()},
            **{f"val_{k}": v for k, v in m_val.items()},
            **{f"test_{k}": v for k, v in m_test.items()},
        }
    )


def run_experiments_cp2(
    train_path: Path = TRAIN_PARQUET_PATH,
    val_path: Path = VAL_PARQUET_PATH,
    test_path: Path = TEST_PARQUET_PATH,
    out_csv: Path = REPORT_TABLES_DIR / "experiments_cp2.csv",
    use_mlflow: bool = True,
) -> pd.DataFrame:
    set_global_seed(SEED)
    ensure_directories()

    if use_mlflow:
        import mlflow

        setup_mlflow("cp2")
        parent_run = mlflow.start_run(run_name="cp2_batch")
    else:
        parent_run = nullcontext()
    with parent_run:
        return _run_experiments_cp2_inner(
            train_path,
            val_path,
            test_path,
            out_csv,
            use_mlflow,
        )


def _run_experiments_cp2_inner(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    out_csv: Path,
    use_mlflow: bool,
) -> pd.DataFrame:
    if use_mlflow:
        import mlflow

    x_tr, y_tr, cols = load_xy(train_path)
    x_va, y_va, _ = load_xy(val_path)
    x_te, y_te, _ = load_xy(test_path)
    log.info("Признаков: %d | train %s val %s test %s", len(cols), x_tr.shape, x_va.shape, x_te.shape)

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    # LogReg L2 / L1
    for pen, tag in (("l2", "logreg_l2_rs"), ("l1", "logreg_l1_rs")):
        with _mlflow_run(use_mlflow, tag):
            model, params, cv_auc = tune_logistic_regression(x_tr, y_tr, penalty=pen)
            m_tr = evaluate_split(model, x_tr, y_tr)
            m_va = evaluate_split(model, x_va, y_va)
            m_te = evaluate_split(model, x_te, y_te)
            if use_mlflow:
                import mlflow

                mlflow.log_params({f"param_{k}": str(v) for k, v in params.items()})
                for split_name, mm in ("train", m_tr), ("val", m_va), ("test", m_te):
                    for k, v in mm.items():
                        mlflow.log_metric(f"{split_name}_{k}", v)
            _append_row(
                rows,
                name=tag,
                hypothesis="Линейная модель с подбором C даёт стабильный бейзлайн относительно CP1 LogReg.",
                model_desc=f"LogisticRegression({pen}) + StandardScaler",
                params=params,
                cv_auc=cv_auc,
                m_train=m_tr,
                m_val=m_va,
                m_test=m_te,
            )
            fitted[tag] = model

    # KNN
    with _mlflow_run(use_mlflow, "knn_rs"):
        model, params, cv_auc = tune_knn(x_tr, y_tr)
        m_tr = evaluate_split(model, x_tr, y_tr)
        m_va = evaluate_split(model, x_va, y_va)
        m_te = evaluate_split(model, x_te, y_te)
        if use_mlflow:
            import mlflow

            mlflow.log_params({f"param_{k}": str(v) for k, v in params.items()})
            for split_name, mm in ("train", m_tr), ("val", m_va), ("test", m_te):
                for k, v in mm.items():
                    mlflow.log_metric(f"{split_name}_{k}", v)
        _append_row(
            rows,
            name="knn_rs",
            hypothesis="KNN на масштабированных признаках хуже GBM при умеренной размерности.",
            model_desc="KNeighborsClassifier + StandardScaler",
            params=params,
            cv_auc=cv_auc,
            m_train=m_tr,
            m_val=m_va,
            m_test=m_te,
        )
        fitted["knn_rs"] = model

    # RandomForest
    with _mlflow_run(use_mlflow, "rf_rs"):
        model, params, cv_auc = tune_random_forest(x_tr, y_tr)
        m_tr = evaluate_split(model, x_tr, y_tr)
        m_va = evaluate_split(model, x_va, y_va)
        m_te = evaluate_split(model, x_te, y_te)
        if use_mlflow:
            import mlflow

            mlflow.log_params({f"param_{k}": str(v) for k, v in params.items()})
            for split_name, mm in ("train", m_tr), ("val", m_va), ("test", m_te):
                for k, v in mm.items():
                    mlflow.log_metric(f"{split_name}_{k}", v)
        _append_row(
            rows,
            name="rf_rs",
            hypothesis="Случайный лес улавливает нелинейности без бустинга.",
            model_desc="RandomForestClassifier",
            params=params,
            cv_auc=cv_auc,
            m_train=m_tr,
            m_val=m_va,
            m_test=m_te,
        )
        fitted["rf_rs"] = model

    boost_runs: list[tuple[Any, str, str, str]] = [
        (
            tune_xgboost_optuna,
            "xgb_optuna",
            "XGBClassifier",
            "XGBoost с Optuna превосходит RF по ROC-AUC на CV.",
        ),
        (
            tune_lightgbm_optuna,
            "lgbm_optuna",
            "LGBMClassifier",
            "LightGBM сопоставим или лучше XGBoost при меньшем времени.",
        ),
    ]
    try:
        import catboost  # noqa: F401

        from src.modeling.tuners import tune_catboost_optuna

        boost_runs.append(
            (
                tune_catboost_optuna,
                "cat_optuna",
                "CatBoostClassifier",
                "CatBoost устойчив к смешанным признакам и конкурирует с LGBM.",
            )
        )
    except ImportError:
        log.warning("CatBoost недоступен, эксперимент cat_optuna пропущен")

    for tuner, tag, desc, hyp in boost_runs:
        with _mlflow_run(use_mlflow, tag):
            model, params, cv_auc = tuner(x_tr, y_tr)
            m_tr = evaluate_split(model, x_tr, y_tr)
            m_va = evaluate_split(model, x_va, y_va)
            m_te = evaluate_split(model, x_te, y_te)
            if use_mlflow:
                mlflow.log_params({f"param_{k}": str(v) for k, v in params.items()})
                for split_name, mm in ("train", m_tr), ("val", m_va), ("test", m_te):
                    for k, v in mm.items():
                        mlflow.log_metric(f"{split_name}_{k}", v)
            _append_row(
                rows,
                name=tag,
                hypothesis=hyp,
                model_desc=desc,
                params=params,
                cv_auc=cv_auc,
                m_train=m_tr,
                m_val=m_va,
                m_test=m_te,
            )
            fitted[tag] = model

    # Calibrated LGBM
    lgbm = fitted["lgbm_optuna"]
    with _mlflow_run(use_mlflow, "lgbm_calibrated"):
        cal = make_calibrated_lgbm(clone(lgbm))
        cal.fit(x_tr, y_tr)
        m_tr = evaluate_split(cal, x_tr, y_tr)
        m_va = evaluate_split(cal, x_va, y_va)
        m_te = evaluate_split(cal, x_te, y_te)
        if use_mlflow:
            import mlflow

            for split_name, mm in ("train", m_tr), ("val", m_va), ("test", m_te):
                for k, v in mm.items():
                    mlflow.log_metric(f"{split_name}_{k}", v)
        _append_row(
            rows,
            name="lgbm_calibrated",
            hypothesis="Калибровка isotonic не снижает ROC-AUC, улучшает вероятности для порогов.",
            model_desc="CalibratedClassifierCV(LGBM, isotonic)",
            params={},
            cv_auc=None,
            m_train=m_tr,
            m_val=m_va,
            m_test=m_te,
        )
        fitted["lgbm_calibrated"] = cal

    # Stacking
    with _mlflow_run(use_mlflow, "stacking_rf_xgb_lgb"):
        est: list[tuple[str, Any]] = [
            ("rf", clone(fitted["rf_rs"])),
            ("xgb", clone(fitted["xgb_optuna"])),
            ("lgb", clone(fitted["lgbm_optuna"])),
        ]
        stack = StackingClassifier(
            estimators=est,
            final_estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("meta", LogisticRegression(max_iter=2000, random_state=SEED)),
                ]
            ),
            cv=3,
            n_jobs=-1,
        )
        stack.fit(x_tr, y_tr)
        m_tr = evaluate_split(stack, x_tr, y_tr)
        m_va = evaluate_split(stack, x_va, y_va)
        m_te = evaluate_split(stack, x_te, y_te)
        if use_mlflow:
            import mlflow

            for split_name, mm in ("train", m_tr), ("val", m_va), ("test", m_te):
                for k, v in mm.items():
                    mlflow.log_metric(f"{split_name}_{k}", v)
        _append_row(
            rows,
            name="stacking_rf_xgb_lgb_meta_logreg",
            hypothesis="Стекинг комбинирует сильные базовые модели и даёт прирост над одиночным LGBM.",
            model_desc="StackingClassifier(RF, XGB, LGBM + LogReg meta)",
            params={"final": "LogisticRegression"},
            cv_auc=None,
            m_train=m_tr,
            m_val=m_va,
            m_test=m_te,
        )
        fitted["stacking"] = stack

    df_out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    log.info("Сохранено %d экспериментов в %s", len(df_out), out_csv)

    return df_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Эксперименты CP2: Optuna/RandomizedSearch, MLflow, CSV.")
    p.add_argument("--train", type=Path, default=TRAIN_PARQUET_PATH)
    p.add_argument("--val", type=Path, default=VAL_PARQUET_PATH)
    p.add_argument("--test", type=Path, default=TEST_PARQUET_PATH)
    p.add_argument("--out-csv", type=Path, default=REPORT_TABLES_DIR / "experiments_cp2.csv")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments_cp2(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        out_csv=args.out_csv,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
