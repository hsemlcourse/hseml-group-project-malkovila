from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED

FAST = os.environ.get("CP2_FAST", "").lower() in ("1", "true", "yes")


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3 if FAST else 5, shuffle=True, random_state=SEED)


def tune_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    penalty: str,
    n_iter: int = 24,
) -> tuple[Pipeline, dict[str, Any], float]:
    if penalty == "l1":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        param_dist = {"clf__C": np.logspace(-3, 2, 20)}
    else:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        param_dist = {"clf__C": np.logspace(-3, 2, 20)}

    rs = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(n_iter, 15 if FAST else n_iter),
        scoring="roc_auc",
        cv=_cv(),
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    rs.fit(x_train, y_train)
    best_score = float(rs.best_score_)
    return rs.best_estimator_, dict(rs.best_params_), best_score


def tune_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 16,
) -> tuple[Pipeline, dict[str, Any], float]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    param_dist = {
        "clf__n_neighbors": [3, 5, 11, 25, 50, 100, 200],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }
    rs = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(n_iter, 10 if FAST else n_iter),
        scoring="roc_auc",
        cv=_cv(),
        random_state=SEED,
        n_jobs=-1,
    )
    rs.fit(x_train, y_train)
    return rs.best_estimator_, dict(rs.best_params_), float(rs.best_score_)


def tune_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 32,
) -> tuple[RandomForestClassifier, dict[str, Any], float]:
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    param_dist = {
        "n_estimators": [200, 400, 800],
        "max_depth": [6, 10, 16, None],
        "min_samples_leaf": [1, 5, 20],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    rs = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=min(n_iter, 12 if FAST else n_iter),
        scoring="roc_auc",
        cv=_cv(),
        random_state=SEED,
        n_jobs=-1,
    )
    rs.fit(x_train, y_train)
    return rs.best_estimator_, dict(rs.best_params_), float(rs.best_score_)


def _tune_xgboost_rs(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Any, dict[str, Any], float]:
    """Fallback без Optuna — RandomizedSearchCV (тот же scoring)."""
    from xgboost import XGBClassifier

    clf = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss")
    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": list(np.logspace(-2.3, -0.7, 8)),
        "subsample": [0.65, 0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.65, 0.75, 0.85, 0.95, 1.0],
        "reg_alpha": list(np.logspace(-4, 1, 6)),
        "reg_lambda": list(np.logspace(-4, 1, 6)),
        "min_child_weight": [1, 3, 5, 9, 15],
    }
    n_iter = 12 if FAST else 35
    rs = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=_cv(),
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    rs.fit(x_train, y_train)
    params = {k: rs.best_params_[k] for k in rs.best_params_}
    params["random_state"] = SEED
    params["n_jobs"] = -1
    params["eval_metric"] = "logloss"
    return rs.best_estimator_, params, float(rs.best_score_)


def _tune_lightgbm_rs(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Any, dict[str, Any], float]:
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1)
    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [3, 5, 7, 9, 11, 12],
        "learning_rate": list(np.logspace(-2.3, -0.7, 8)),
        "subsample": [0.65, 0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.65, 0.75, 0.85, 0.95, 1.0],
        "reg_alpha": list(np.logspace(-4, 1, 6)),
        "reg_lambda": list(np.logspace(-4, 1, 6)),
        "num_leaves": [31, 64, 127, 255],
    }
    n_iter = 12 if FAST else 35
    rs = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=_cv(),
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    rs.fit(x_train, y_train)
    params = {k: rs.best_params_[k] for k in rs.best_params_}
    params["random_state"] = SEED
    params["n_jobs"] = -1
    params["verbose"] = -1
    return rs.best_estimator_, params, float(rs.best_score_)


def _tune_catboost_rs(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Any, dict[str, Any], float]:
    from catboost import CatBoostClassifier

    clf = CatBoostClassifier(random_seed=SEED, verbose=False, thread_count=-1)
    param_dist = {
        "iterations": [300, 500, 700, 1000],
        "depth": [4, 6, 8, 10],
        "learning_rate": list(np.logspace(-2.3, -0.7, 8)),
        "l2_leaf_reg": [1.0, 3.0, 6.0, 10.0],
    }
    n_iter = 10 if FAST else 30
    rs = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=_cv(),
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    rs.fit(x_train, y_train)
    params = {k: rs.best_params_[k] for k in rs.best_params_}
    params["random_seed"] = SEED
    params["verbose"] = False
    params["thread_count"] = -1
    return rs.best_estimator_, params, float(rs.best_score_)


def tune_xgboost_optuna(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any], float]:
    try:
        import optuna
    except ImportError:
        return _tune_xgboost_rs(x_train, y_train)

    from xgboost import XGBClassifier

    trials = n_trials or (8 if FAST else 50)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "random_state": SEED,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        clf = XGBClassifier(**params)
        scores: list[float] = []
        for train_idx, val_idx in _cv().split(x_train, y_train):
            clf.fit(x_train[train_idx], y_train[train_idx])
            proba = clf.predict_proba(x_train[val_idx])[:, 1]
            scores.append(roc_auc_score(y_train[val_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = study.best_params.copy()
    best_params["random_state"] = SEED
    best_params["n_jobs"] = -1
    best_params["eval_metric"] = "logloss"
    model = XGBClassifier(**best_params)
    model.fit(x_train, y_train)
    return model, best_params, float(study.best_value)


def tune_lightgbm_optuna(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any], float]:
    try:
        import optuna
    except ImportError:
        return _tune_lightgbm_rs(x_train, y_train)

    from lightgbm import LGBMClassifier

    trials = n_trials or (8 if FAST else 50)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        clf = LGBMClassifier(**params)
        scores: list[float] = []
        for train_idx, val_idx in _cv().split(x_train, y_train):
            clf.fit(x_train[train_idx], y_train[train_idx])
            proba = clf.predict_proba(x_train[val_idx])[:, 1]
            scores.append(roc_auc_score(y_train[val_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = study.best_params.copy()
    best_params["random_state"] = SEED
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1
    model = LGBMClassifier(**best_params)
    model.fit(x_train, y_train)
    return model, best_params, float(study.best_value)


def tune_catboost_optuna(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any], float]:
    try:
        import optuna
    except ImportError:
        try:
            return _tune_catboost_rs(x_train, y_train)
        except ImportError as exc:  # catboost
            raise ImportError("Нужны optuna или catboost") from exc

    from catboost import CatBoostClassifier

    trials = n_trials or (6 if FAST else 40)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1200),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_seed": SEED,
            "verbose": False,
            "thread_count": -1,
        }
        clf = CatBoostClassifier(**params)
        scores: list[float] = []
        for train_idx, val_idx in _cv().split(x_train, y_train):
            clf.fit(x_train[train_idx], y_train[train_idx])
            proba = clf.predict_proba(x_train[val_idx])[:, 1]
            scores.append(roc_auc_score(y_train[val_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = study.best_params.copy()
    best_params["random_seed"] = SEED
    best_params["verbose"] = False
    best_params["thread_count"] = -1
    model = CatBoostClassifier(**best_params)
    model.fit(x_train, y_train)
    return model, best_params, float(study.best_value)


def make_calibrated_lgbm(lgbm: Any) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
