"""
Phase 3 step 3 - FAIR RE-EVALUATION via MONTH-BLOCKED (leave-one-month-out) CV.

Motivation
----------
The single Jan-Sep / Oct-Dec split (step 2) tested the dust feature on a
dust-free quarter (Oct-Dec max dust AOD 0.29; zero of the 3,504 annual >0.4413
high-dust timesteps landed in test), so it could not measure any dust benefit.

This step changes ONLY the evaluation protocol. Features, models,
hyperparameters, target, daytime rule and seeds are reused verbatim from
src/features.py and src/modeling.py.

CV design (leave-one-month-out, 12 folds)
-----------------------------------------
* Fold k (k = 1..12): calendar month k is the held-out TEST month; the other 11
  months are TRAIN. Every dust-heavy month (Mar-Aug) is therefore held out once,
  so the dust feature is evaluated on real dust events.
* Early stopping: within each fold, the validation slice is the LAST 14 DAYS of
  the chronologically latest training month (Dec, or Nov when Dec is the test
  month). Identical for baseline and dust-aware.
* Daytime-only evaluation (solar zenith < 90 deg at target time), as before.
* Both models (LightGBM, XGBoost) and both variants (baseline dust-blind vs
  dust-aware with time-t dust features) run through the identical pipeline.

Pooled reporting: because months partition the year, concatenating each fold's
daytime test predictions reconstructs a full-year daytime prediction set in which
every row was held out exactly once.

Run:
    .venv\\Scripts\\python.exe src\\train_dust_aware_cv.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features as F
import modeling as M

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Annual high-dust threshold from the data gate (90th pct of cams_dust_aod550,
# full-year 15-min series). Reused unchanged so the slice means "real dust".
ANNUAL_HIGH_DUST_THR = 0.4413

VARIANTS = {
    "baseline": F.BASE_FEATURES,                       # dust-blind
    "dust_aware": F.BASE_FEATURES + F.DUST_T_FEATURES,  # + time-t dust (main)
}
MODELS = ["LightGBM", "XGBoost"]
VALID_DAYS = 14  # last ~2 weeks of latest training month -> early-stopping slice


def _fold_valid_split(train_all: pd.DataFrame):
    """Validation = last VALID_DAYS of the chronologically latest training month.
    Returns (train_fit, valid). Identical regardless of feature variant."""
    latest_month = max(train_all.index.month.unique())
    mrows = train_all[train_all.index.month == latest_month]
    cutoff = mrows.index.max() - pd.Timedelta(days=VALID_DAYS)
    valid = mrows[mrows.index >= cutoff]
    train_fit = train_all.drop(valid.index)
    return train_fit, valid


def _fit_predict(model_name, cols, train_fit, valid, X):
    Xtr, ytr = train_fit[cols], train_fit["y"]
    Xva, yva = valid[cols], valid["y"]
    if model_name == "LightGBM":
        mdl = M.fit_lgbm(M.make_lgbm(), Xtr, ytr, Xva, yva)
    else:
        mdl = M.fit_xgb(M.make_xgb(), Xtr, ytr, Xva, yva)
    return M.predict_clip(mdl, X[cols])


def main() -> int:
    df = F.load_merged()
    feat = F.build_features(df)

    # Daytime rows (each will be held out exactly once across the 12 folds).
    dd = feat[feat["daytime"]].copy()
    day_index = dd.index

    # Prediction containers indexed by the daytime rows.
    pred = {(v, m): pd.Series(index=day_index, dtype=float)
            for v in VARIANTS for m in MODELS}

    months = list(range(1, 13))
    print("Running 12 leave-one-month-out folds (2 variants x 2 models each)...")
    for test_month in months:
        test = feat[feat.index.month == test_month]
        test_day = test[test["daytime"]]
        train_all = feat[feat.index.month != test_month]
        train_fit, valid = _fold_valid_split(train_all)
        for variant, cols in VARIANTS.items():
            for model_name in MODELS:
                p = _fit_predict(model_name, cols, train_fit, valid, test_day)
                pred[(variant, model_name)].loc[test_day.index] = p
        print(f"  fold {test_month:2d}: test_day={len(test_day):5d} "
              f"train_fit={len(train_fit)} valid={len(valid)} "
              f"(valid month={max(train_all.index.month.unique())})")

    # ---- Pooled arrays ----------------------------------------------------- #
    y = dd["y"].to_numpy()
    smart = np.clip(dd["pred_smart_persist"].to_numpy(), 0, None)
    plain = np.clip(dd["pred_plain_persist"].to_numpy(), 0, None)
    dust_target = dd[F.DUST_SLICE_COL].to_numpy()  # dust AOD at t+1h
    month_arr = dd.index.month.to_numpy()

    hd_mask = dust_target > ANNUAL_HIGH_DUST_THR
    slices = {"all-sky": np.ones(len(dd), bool), "high-dust": hd_mask}
    mean_ghi = {s: float(y[m].mean()) for s, m in slices.items()}
    rmse_ref = {s: M.rmse(y[m], smart[m]) for s, m in slices.items()}

    # ---- Pooled metrics table --------------------------------------------- #
    rows = []

    def _emit(model, variant, p):
        for s, m in slices.items():
            met = M.metrics(y[m], p[m], mean_ghi[s], rmse_ref[s])
            rows.append({"model": model, "variant": variant, "slice": s, **met})

    _emit("SmartPersistence", "-", smart)
    _emit("PlainPersistence", "-", plain)
    for variant in VARIANTS:
        for model_name in MODELS:
            _emit(model_name, variant, pred[(variant, model_name)].to_numpy())

    table = pd.DataFrame(rows).set_index(["model", "variant", "slice"])
    table = table[["RMSE", "MAE", "MBE", "nRMSE", "R2", "skill_vs_smart_persist"]]
    table.to_csv(RESULTS / "cv_ablation_metrics.csv", float_format="%.4f")

    # ---- Improvement (dust-aware vs baseline), all-sky vs high-dust -------- #
    imp_rows = []
    for model_name in MODELS:
        for s in slices:
            b = table.loc[(model_name, "baseline", s), "RMSE"]
            d = table.loc[(model_name, "dust_aware", s), "RMSE"]
            imp_rows.append({"model": model_name, "slice": s,
                             "RMSE_baseline": b, "RMSE_dust_aware": d,
                             "RMSE_improve_abs": b - d,
                             "RMSE_improve_pct": 100.0 * (b - d) / b})
    improve = pd.DataFrame(imp_rows).set_index(["model", "slice"])

    # ---- Per-month breakdown ---------------------------------------------- #
    pm_rows = []
    for mo in months:
        sel = month_arr == mo
        ym = y[sel]
        rec = {"month": mo, "n_daytime": int(sel.sum()),
               "dust_mean": float(dust_target[sel].mean()),
               "dust_max": float(dust_target[sel].max()),
               "n_highdust": int((dust_target[sel] > ANNUAL_HIGH_DUST_THR).sum())}
        for model_name in MODELS:
            b = M.rmse(ym, pred[("baseline", model_name)].to_numpy()[sel])
            d = M.rmse(ym, pred[("dust_aware", model_name)].to_numpy()[sel])
            rec[f"{model_name}_baseline_RMSE"] = b
            rec[f"{model_name}_dust_RMSE"] = d
            rec[f"{model_name}_improve_pct"] = 100.0 * (b - d) / b
        pm_rows.append(rec)
    permonth = pd.DataFrame(pm_rows).set_index("month")
    permonth.to_csv(RESULTS / "cv_permonth.csv", float_format="%.4f")

    # ---- Figures ----------------------------------------------------------- #
    _plot_permonth(permonth)
    _plot_improvement_bars(improve)

    # ---- Summary ----------------------------------------------------------- #
    n_hd = int(hd_mask.sum())
    print("=" * 80)
    print("MONTH-BLOCKED CV ABLATION  (leave-one-month-out, 12 folds, daytime)")
    print("=" * 80)
    print(f"Pooled daytime test rows (each held out once): {len(dd)}")
    print(f"High-dust slice: dust AOD(t+1h) > {ANNUAL_HIGH_DUST_THR} (annual data-gate threshold)")
    print(f"  pooled high-dust daytime test rows: {n_hd} "
          f"({100.0*n_hd/len(dd):.1f}% of daytime) -- real dust events now present")
    print(f"  mean daytime GHI: all-sky={mean_ghi['all-sky']:.1f} | "
          f"high-dust={mean_ghi['high-dust']:.1f} W/m^2")
    print(f"  smart-persist RMSE ref: all-sky={rmse_ref['all-sky']:.2f} | "
          f"high-dust={rmse_ref['high-dust']:.2f} W/m^2")
    print("-" * 80)
    print("POOLED METRICS (baseline vs dust-aware, both models):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}",
                           "display.width", 200):
        print(table.to_string())
    print("-" * 80)
    print("RMSE IMPROVEMENT dust-aware vs baseline (+ = dust helps):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}",
                           "display.width", 200):
        print(improve.to_string())
    print("-" * 80)
    print("PER-MONTH (RMSE baseline -> dust-aware; dusty months in bold-ish):")
    show = permonth[["dust_mean", "dust_max", "n_highdust",
                     "LightGBM_baseline_RMSE", "LightGBM_dust_RMSE", "LightGBM_improve_pct",
                     "XGBoost_baseline_RMSE", "XGBoost_dust_RMSE", "XGBoost_improve_pct"]]
    with pd.option_context("display.float_format", lambda v: f"{v:7.2f}",
                           "display.width", 220):
        print(show.to_string())
    print("-" * 80)
    print("Saved: results/cv_ablation_metrics.csv, results/cv_permonth.csv")
    print("Figures: figures/cv_permonth_rmse.png, figures/cv_improvement_bars.png")
    return 0


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _plot_permonth(pm: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(12)
    w = 0.38
    for ax, model in zip(axes, MODELS):
        base = pm[f"{model}_baseline_RMSE"].to_numpy()
        dust = pm[f"{model}_dust_RMSE"].to_numpy()
        ax.bar(x - w / 2, base, w, label="baseline (dust-blind)", color="#94a3b8")
        ax.bar(x + w / 2, dust, w, label="dust-aware (time-t)", color="#b45309")
        ax.set_ylabel(f"{model} RMSE [W/m$^2$]")
        ax.legend(loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(x, pm["dust_mean"].to_numpy(), color="#1e3a8a", marker="o",
                 lw=1.5, label="mean dust AOD")
        ax2.plot(x, pm["dust_max"].to_numpy(), color="#1e3a8a", marker=".",
                 lw=1.0, ls="--", alpha=0.6, label="max dust AOD")
        ax2.axhline(ANNUAL_HIGH_DUST_THR, color="#7f1d1d", lw=1, ls=":",
                    label=f"high-dust thr {ANNUAL_HIGH_DUST_THR}")
        ax2.set_ylabel("dust AOD 550nm")
        ax2.legend(loc="upper right", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MONTH_LABELS)
    axes[0].set_title("Per-month RMSE: baseline vs dust-aware, with dust-AOD overlay\n"
                      "(leave-one-month-out CV; dusty months Mar-Aug)")
    fig.tight_layout()
    fig.savefig(FIGURES / "cv_permonth_rmse.png", dpi=120)
    plt.close(fig)


def _plot_improvement_bars(improve: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS))
    w = 0.35
    allsky = [improve.loc[(m, "all-sky"), "RMSE_improve_pct"] for m in MODELS]
    highd = [improve.loc[(m, "high-dust"), "RMSE_improve_pct"] for m in MODELS]
    b1 = ax.bar(x - w / 2, allsky, w, label="all-sky", color="#94a3b8")
    b2 = ax.bar(x + w / 2, highd, w, label="high-dust (>0.4413)", color="#b45309")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():+.1f}%",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom" if r.get_height() >= 0 else "top",
                        fontsize=9)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("RMSE improvement from dust [%]  (+ = dust helps)")
    ax.set_title("Dust-aware vs baseline RMSE improvement (pooled CV)\nall-sky vs true high-dust slice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cv_improvement_bars.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
