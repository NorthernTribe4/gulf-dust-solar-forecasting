"""
Shared modelling helpers for the baseline and dust-aware runs.

Centralising the model constructors, fit routines and metrics here guarantees the
dust-aware run uses byte-identical hyperparameters, seeds and early-stopping to
the baseline -- the ONLY intended difference between the two is the feature list.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from xgboost import XGBRegressor

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50


def metrics(y_true, y_pred, mean_ghi, rmse_ref) -> dict:
    """RMSE/MAE/MBE/nRMSE/R2 plus skill vs a reference RMSE (smart persistence)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(err))
    nrmse = float(rmse / mean_ghi) if mean_ghi else np.nan
    r2 = float(r2_score(y_true, y_pred))
    skill = float(1.0 - rmse / rmse_ref) if rmse_ref > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "MBE": mbe,
            "nRMSE": nrmse, "R2": r2, "skill_vs_smart_persist": skill}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2)))


# --------------------------------------------------------------------------- #
# Model factories (identical settings for every run)
# --------------------------------------------------------------------------- #
def make_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )


def make_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric="rmse",
    )


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE))


def fit_lgbm(model, Xtr, ytr, Xva, yva):
    model.fit(
        Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


def fit_xgb(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


def predict_clip(model, X) -> np.ndarray:
    """Predict and clip to physically valid GHI (>= 0)."""
    return np.clip(model.predict(X), 0, None)
