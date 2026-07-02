"""
Phase 3 step 2 - DUST-AWARE model + ABLATION (the core experiment).

Controlled comparison: reuses src/features.py (identical split, daytime rule,
target, warm-up) and src/modeling.py (identical models, hyperparameters, seeds,
early stopping). The ONLY difference between the baseline and the dust-aware run
is the appended CAMS dust-AOD feature(s). Baseline and dust-aware are trained and
evaluated inside THIS one script so the comparison is guaranteed apples-to-apples.

Variants
--------
  baseline       : BASE_FEATURES only (dust-blind)              [reference]
  dust_aware     : BASE + time-t dust AOD (lag0, lag4, roll4)   [MAIN result]
  dust_forecast  : dust_aware + dust AOD at t+1h (FUTURE info)  [secondary]
  total_aod      : BASE + time-t TOTAL AOD (not dust-specific)  [secondary]

Slices (daytime Oct-Dec hold-out)
  all-sky   : every daytime test row
  high-dust : daytime test rows with dust AOD(t+1h) > 90th pct of the test slice

Run:
    .venv\\Scripts\\python.exe src\\train_dust_aware.py
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

# Feature sets: name -> ordered feature column list. Dust cols appended to BASE.
VARIANTS = {
    "baseline": F.BASE_FEATURES,
    "dust_aware": F.BASE_FEATURES + F.DUST_T_FEATURES,               # MAIN
    "dust_forecast": F.BASE_FEATURES + F.DUST_T_FEATURES + F.DUST_FORECAST_FEATURES,
    "total_aod": F.BASE_FEATURES + F.TOTAL_T_FEATURES,
}
MAIN_VARIANTS = ["baseline", "dust_aware"]          # headline ablation
SECONDARY_VARIANTS = ["dust_forecast", "total_aod"]
MODELS = ["LightGBM", "XGBoost"]


def _fit_predict(model_name, cols, train_fit, valid, Xte):
    """Fit one model on train_fit (early-stop on valid) and predict on test."""
    Xtr, ytr = train_fit[cols], train_fit["y"]
    Xva, yva = valid[cols], valid["y"]
    if model_name == "LightGBM":
        mdl = M.fit_lgbm(M.make_lgbm(), Xtr, ytr, Xva, yva)
    else:
        mdl = M.fit_xgb(M.make_xgb(), Xtr, ytr, Xva, yva)
    return mdl, M.predict_clip(mdl, Xte[cols])


def _gain_importance(model_name, mdl, cols) -> pd.Series:
    if model_name == "LightGBM":
        imp = mdl.booster_.feature_importance(importance_type="gain")
        names = mdl.booster_.feature_name()
        s = pd.Series(imp, index=names)
    else:
        score = mdl.get_booster().get_score(importance_type="gain")  # sparse dict
        s = pd.Series({c: score.get(c, 0.0) for c in cols})
    return s.reindex(cols).fillna(0.0)


def main() -> int:
    df = F.load_merged()
    feat = F.build_features(df)
    train_fit, valid, test, train_full = F.chrono_split(feat)

    test_day = test[test["daytime"]].copy()
    yte = test_day["y"].to_numpy()

    # ---- High-dust slice: 90th pct of dust AOD(t+1h) on the daytime test set  #
    p90 = float(test_day[F.DUST_SLICE_COL].quantile(0.90))
    hd_mask = (test_day[F.DUST_SLICE_COL] > p90).to_numpy()
    slices = {"all-sky": np.ones(len(test_day), bool), "high-dust": hd_mask}
    mean_ghi = {s: float(yte[m].mean()) for s, m in slices.items()}

    # ---- Smart-persistence reference (per slice) --------------------------- #
    smart = np.clip(test_day["pred_smart_persist"].to_numpy(), 0, None)
    plain = np.clip(test_day["pred_plain_persist"].to_numpy(), 0, None)
    rmse_ref = {s: M.rmse(yte[m], smart[m]) for s, m in slices.items()}

    rows = []  # metric rows for the CSV/table

    def _emit(model, variant, pred):
        for s, m in slices.items():
            met = M.metrics(yte[m], pred[m], mean_ghi[s], rmse_ref[s])
            rows.append({"model": model, "variant": variant, "slice": s, **met})

    # persistence references (variant-independent)
    _emit("SmartPersistence", "-", smart)
    _emit("PlainPersistence", "-", plain)

    # ---- Train every variant x model through the identical pipeline -------- #
    preds: dict = {}
    fitted: dict = {}
    for variant, cols in VARIANTS.items():
        preds[variant] = {}
        fitted[variant] = {}
        for model_name in MODELS:
            mdl, p = _fit_predict(model_name, cols, train_fit, valid, test_day)
            preds[variant][model_name] = p
            fitted[variant][model_name] = mdl
            _emit(model_name, variant, p)

    table = pd.DataFrame(rows).set_index(["model", "variant", "slice"])
    table = table[["RMSE", "MAE", "MBE", "nRMSE", "R2", "skill_vs_smart_persist"]]

    # ---- RMSE improvement: dust-aware vs baseline, per model per slice ------ #
    improve_rows = []
    for model_name in MODELS:
        for s in slices:
            base_rmse = table.loc[(model_name, "baseline", s), "RMSE"]
            for variant in ["dust_aware", "dust_forecast", "total_aod"]:
                v_rmse = table.loc[(model_name, variant, s), "RMSE"]
                improve_rows.append({
                    "model": model_name, "variant_vs_baseline": variant, "slice": s,
                    "RMSE_baseline": base_rmse, "RMSE_variant": v_rmse,
                    "RMSE_improve_abs": base_rmse - v_rmse,
                    "RMSE_improve_pct": 100.0 * (base_rmse - v_rmse) / base_rmse,
                })
    improve = pd.DataFrame(improve_rows).set_index(["model", "variant_vs_baseline", "slice"])

    # ---- Save metrics ------------------------------------------------------ #
    metrics_path = RESULTS / "ablation_metrics.csv"
    table.to_csv(metrics_path, float_format="%.4f")
    improve_path = RESULTS / "ablation_rmse_improvement.csv"
    improve.to_csv(improve_path, float_format="%.4f")

    # ---- Feature importance (dust-aware, both models) ---------------------- #
    imp_lgbm = _gain_importance("LightGBM", fitted["dust_aware"]["LightGBM"],
                                VARIANTS["dust_aware"])
    imp_xgb = _gain_importance("XGBoost", fitted["dust_aware"]["XGBoost"],
                               VARIANTS["dust_aware"])
    dust_names = set(F.DUST_T_FEATURES)

    def _ranks(imp: pd.Series) -> dict:
        order = imp.sort_values(ascending=False)
        rank = {name: (int(list(order.index).index(name)) + 1) for name in F.DUST_T_FEATURES}
        return rank

    rank_lgbm = _ranks(imp_lgbm)
    rank_xgb = _ranks(imp_xgb)

    # ---- Plots ------------------------------------------------------------- #
    _plot_rmse_bars(table)
    _plot_importance(imp_lgbm, dust_names)
    _plot_highdust_pred(yte, preds, hd_mask)

    # ---- Print summary ----------------------------------------------------- #
    n_all = int(slices["all-sky"].sum())
    n_hd = int(slices["high-dust"].sum())
    print("=" * 78)
    print("DUST-AWARE ABLATION  (1-hour-ahead GHI, daytime Oct-Dec 2022 hold-out)")
    print("=" * 78)
    print(f"Controlled comparison: baseline vs dust-aware differ ONLY by dust features.")
    print(f"  baseline features   : {len(F.BASE_FEATURES)} (dust-blind)")
    print(f"  dust_aware adds      : {F.DUST_T_FEATURES}  (all KNOWN at issue time t)")
    print(f"  dust_forecast adds   : {F.DUST_FORECAST_FEATURES}  (dust AOD at t+1h = FUTURE)")
    print(f"  total_aod swaps in   : {F.TOTAL_T_FEATURES}  (time-t, not dust-specific)")
    print("-" * 78)
    print(f"High-dust slice: dust AOD(t+1h) > 90th pct = {p90:.4f}")
    print(f"  daytime test rows: all-sky={n_all} | high-dust={n_hd} "
          f"({100.0*n_hd/n_all:.1f}% of daytime test)")
    print(f"  mean daytime GHI : all-sky={mean_ghi['all-sky']:.1f} | "
          f"high-dust={mean_ghi['high-dust']:.1f} W/m^2")
    print(f"  smart-persist RMSE ref: all-sky={rmse_ref['all-sky']:.2f} | "
          f"high-dust={rmse_ref['high-dust']:.2f} W/m^2")
    print("-" * 78)
    print("FULL METRICS TABLE:")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}",
                           "display.width", 200):
        print(table.to_string())
    print("-" * 78)
    print("RMSE IMPROVEMENT vs baseline (positive = dust helps):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}",
                           "display.width", 200):
        print(improve.to_string())
    print("-" * 78)
    print("DUST FEATURE IMPORTANCE RANK (dust_aware model, gain; 1 = most important):")
    print(f"  total features in dust_aware: {len(VARIANTS['dust_aware'])}")
    print(f"  LightGBM: " + ", ".join(f"{k}=#{v}" for k, v in rank_lgbm.items()))
    print(f"  XGBoost : " + ", ".join(f"{k}=#{v}" for k, v in rank_xgb.items()))
    print("  LightGBM top-8 by gain:")
    print(imp_lgbm.sort_values(ascending=False).head(8).to_string())
    print("-" * 78)
    print(f"Saved: {metrics_path.relative_to(ROOT)}, {improve_path.relative_to(ROOT)}")
    print("Figures: figures/ablation_rmse_bars.png, ablation_feature_importance.png, "
          "ablation_highdust_pred.png")
    return 0


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plot_rmse_bars(table: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    slices = ["all-sky", "high-dust"]
    x = np.arange(len(slices))
    w = 0.35
    for ax, model in zip(axes, MODELS):
        base = [table.loc[(model, "baseline", s), "RMSE"] for s in slices]
        dust = [table.loc[(model, "dust_aware", s), "RMSE"] for s in slices]
        b1 = ax.bar(x - w / 2, base, w, label="baseline (dust-blind)", color="#94a3b8")
        b2 = ax.bar(x + w / 2, dust, w, label="dust-aware (time-t)", color="#b45309")
        for bars in (b1, b2):
            for r in bars:
                ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=9)
        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels(slices)
        ax.set_ylabel("RMSE [W/m$^2$]")
        ax.legend()
    fig.suptitle("Baseline vs dust-aware RMSE - all-sky vs high-dust (daytime test)")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_rmse_bars.png", dpi=120)
    plt.close(fig)


def _plot_importance(imp: pd.Series, dust_names: set):
    top = imp.sort_values(ascending=False).head(15)[::-1]
    colors = ["#b45309" if n in dust_names else "#64748b" for n in top.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color=colors)
    ax.set_title("Dust-aware LightGBM - top-15 feature importance (gain)\n"
                 "(dust features highlighted)")
    ax.set_xlabel("gain")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_feature_importance.png", dpi=120)
    plt.close(fig)


def _plot_highdust_pred(yte, preds, hd_mask):
    y = yte[hd_mask]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    hi = max(y.max(),
             preds["baseline"]["LightGBM"][hd_mask].max(),
             preds["dust_aware"]["LightGBM"][hd_mask].max()) * 1.02
    for ax, variant, title in zip(
            axes, ["baseline", "dust_aware"],
            ["baseline (dust-blind)", "dust-aware (time-t)"]):
        p = preds[variant]["LightGBM"][hd_mask]
        ax.scatter(y, p, s=10, alpha=0.4, color="#7c2d12")
        ax.plot([0, hi], [0, hi], color="#334155", lw=1, ls="--")
        r = np.sqrt(np.mean((p - y) ** 2))
        ax.set_title(f"LightGBM {title}\nhigh-dust RMSE={r:.1f} W/m$^2$")
        ax.set_xlabel("Actual GHI(t+1h) [W/m$^2$]")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
    axes[0].set_ylabel("Predicted GHI(t+1h) [W/m$^2$]")
    fig.suptitle("High-dust days: predicted vs actual (LightGBM)")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_highdust_pred.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
