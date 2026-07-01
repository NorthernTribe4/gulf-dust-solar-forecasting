"""
Phase 3 step 1 - DUST-BLIND baseline GHI forecaster.

Trains LightGBM, XGBoost and a Ridge linear reference to predict GHI 1 hour
ahead using only non-aerosol features (see src/features.py). Evaluates on a
chronological Oct-Dec hold-out, DAYTIME timesteps only, against smart- and
plain-persistence references, and reports a forecast skill score.

NO aerosol/dust feature is used here; this is the reference the dust-aware model
must beat. This script builds/evaluates only -- it does not tune.

Run:
    .venv\\Scripts\\python.exe src\\train_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from xgboost import XGBRegressor

import features as F

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(y_true, y_pred, mean_ghi, rmse_ref):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(err))
    nrmse = float(rmse / mean_ghi)
    r2 = float(r2_score(y_true, y_pred))
    skill = float(1.0 - rmse / rmse_ref) if rmse_ref > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "MBE": mbe,
            "nRMSE": nrmse, "R2": r2, "skill_vs_smart_persist": skill}


def main() -> int:
    # ---- Build design frame ------------------------------------------------ #
    df = F.load_merged()
    feat = F.build_features(df)
    feat_cols = F.base_feature_columns(feat)

    train_fit, valid, test, train_full = F.chrono_split(feat)

    # Daytime-only evaluation set (sun up at target time).
    test_day = test[test["daytime"]]
    train_day = train_full[train_full["daytime"]]

    Xtr, ytr = train_fit[feat_cols], train_fit["y"]
    Xva, yva = valid[feat_cols], valid["y"]
    Xtr_full, ytr_full = train_full[feat_cols], train_full["y"]
    Xte, yte = test_day[feat_cols], test_day["y"]

    mean_ghi_test = float(test_day["y"].mean())

    # ---- Reference forecasts (evaluate on daytime test) -------------------- #
    smart = np.clip(test_day["pred_smart_persist"].to_numpy(), 0, None)
    plain = np.clip(test_day["pred_plain_persist"].to_numpy(), 0, None)
    rmse_smart = float(np.sqrt(np.mean((smart - yte.to_numpy()) ** 2)))

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    results["SmartPersistence"] = _metrics(yte.to_numpy(), smart, mean_ghi_test, rmse_smart)
    results["PlainPersistence"] = _metrics(yte.to_numpy(), plain, mean_ghi_test, rmse_smart)
    preds["SmartPersistence"] = smart
    preds["PlainPersistence"] = plain

    # ---- Ridge linear reference -------------------------------------------- #
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ridge.fit(Xtr_full, ytr_full)  # linear: no early stopping, use full train
    p_ridge = np.clip(ridge.predict(Xte), 0, None)
    results["LinearRidge"] = _metrics(yte.to_numpy(), p_ridge, mean_ghi_test, rmse_smart)
    preds["LinearRidge"] = p_ridge

    # ---- LightGBM ---------------------------------------------------------- #
    lgbm = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)], eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    p_lgbm = np.clip(lgbm.predict(Xte), 0, None)
    results["LightGBM"] = _metrics(yte.to_numpy(), p_lgbm, mean_ghi_test, rmse_smart)
    preds["LightGBM"] = p_lgbm
    lgbm_best_iter = lgbm.best_iteration_

    # ---- XGBoost ----------------------------------------------------------- #
    xgb = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=50,
        eval_metric="rmse",
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p_xgb = np.clip(xgb.predict(Xte), 0, None)
    results["XGBoost"] = _metrics(yte.to_numpy(), p_xgb, mean_ghi_test, rmse_smart)
    preds["XGBoost"] = p_xgb
    xgb_best_iter = xgb.best_iteration

    # ---- Metrics table ----------------------------------------------------- #
    order = ["SmartPersistence", "PlainPersistence", "LinearRidge", "LightGBM", "XGBoost"]
    table = pd.DataFrame({m: results[m] for m in order}).T
    table = table[["RMSE", "MAE", "MBE", "nRMSE", "R2", "skill_vs_smart_persist"]]
    metrics_path = RESULTS / "baseline_metrics.csv"
    table.to_csv(metrics_path, float_format="%.4f")

    # ---- Plots ------------------------------------------------------------- #
    _plot_pred_vs_actual(yte.to_numpy(), preds, mean_ghi_test)
    _plot_residuals(yte.to_numpy(), preds["LightGBM"], test_day.index)

    # ---- Summary ----------------------------------------------------------- #
    print("=" * 72)
    print("DUST-BLIND BASELINE GHI FORECASTER  (1-hour-ahead, 15-min data)")
    print("=" * 72)
    print(f"Site            : Doha (lat={F.SITE_LAT}, lon={F.SITE_LON})")
    print(f"Horizon         : {F.HORIZON_STEPS} steps = {F.HORIZON_STEPS*F.STEP_MIN} min ahead")
    print(f"Features ({len(feat_cols)})   : dust-blind (no nsrdb_aod / cams_* / dni / dhi)")
    print("   " + ", ".join(feat_cols))
    print("-" * 72)
    print("Chronological split (by issue time t, UTC):")
    print(f"  train_fit : {train_fit.index.min()}  ->  {train_fit.index.max()}  (Jan-Aug)")
    print(f"  valid     : {valid.index.min()}  ->  {valid.index.max()}  (Sep, early stop)")
    print(f"  test      : {test.index.min()}  ->  {test.index.max()}  (Oct-Dec, HOLD-OUT)")
    print("Daytime = solar zenith < 90 deg at target time.")
    print(f"  rows: train_full={len(train_full)} (daytime {len(train_day)}) | "
          f"test={len(test)} (daytime {len(test_day)})")
    print(f"  mean daytime test GHI = {mean_ghi_test:.2f} W/m^2")
    print(f"  LightGBM best_iter={lgbm_best_iter} | XGBoost best_iter={xgb_best_iter}")
    print("-" * 72)
    print("METRICS (daytime test set; skill = 1 - RMSE/RMSE_smart_persist):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}"):
        print(table.to_string())
    print("-" * 72)
    print(f"Saved metrics -> {metrics_path.relative_to(ROOT)}")
    print(f"Saved figures -> figures/baseline_pred_vs_actual.png, figures/baseline_residuals.png")
    return 0


def _plot_pred_vs_actual(y_true, preds, mean_ghi):
    models = ["SmartPersistence", "LightGBM", "XGBoost"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    hi = max(y_true.max(), max(p.max() for p in preds.values())) * 1.02
    for ax, m in zip(axes, models):
        ax.scatter(y_true, preds[m], s=3, alpha=0.15, color="#7c2d12")
        ax.plot([0, hi], [0, hi], color="#334155", lw=1, ls="--")
        rmse = np.sqrt(np.mean((preds[m] - y_true) ** 2))
        ax.set_title(f"{m}\nRMSE={rmse:.1f} W/m$^2$")
        ax.set_xlabel("Actual GHI(t+1h) [W/m$^2$]")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
    axes[0].set_ylabel("Predicted GHI(t+1h) [W/m$^2$]")
    fig.suptitle("Dust-blind baseline: predicted vs actual (daytime test, Oct-Dec 2022)")
    fig.tight_layout()
    p = FIGURES / "baseline_pred_vs_actual.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)


def _plot_residuals(y_true, y_pred, index):
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, resid, s=3, alpha=0.15, color="#b45309")
    axes[0].axhline(0, color="#334155", lw=1, ls="--")
    axes[0].set_title("LightGBM residuals vs actual")
    axes[0].set_xlabel("Actual GHI(t+1h) [W/m$^2$]")
    axes[0].set_ylabel("Residual (pred - actual) [W/m$^2$]")
    axes[1].hist(resid, bins=60, color="#b45309", alpha=0.85)
    axes[1].axvline(0, color="#334155", lw=1, ls="--")
    axes[1].set_title(f"LightGBM residual distribution\nMBE={resid.mean():.1f}, "
                      f"std={resid.std():.1f} W/m$^2$")
    axes[1].set_xlabel("Residual (pred - actual) [W/m$^2$]")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    p = FIGURES / "baseline_residuals.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
