"""
Phase 3 step 4 - HORIZON SWEEP (1 h / 3 h / 6 h) for the dust-aware ablation.

Hypothesis to test honestly: at 1 h the observed clear-sky index kt already
captures sky attenuation (incl. dust), so explicit dust AOD is redundant. As the
horizon lengthens, kt-persistence decays and an explicit dust feature MIGHT start
to add value. This sweep measures the baseline-vs-dust-aware ablation across
horizons under the SAME month-blocked leave-one-month-out CV as step 3.

Reuse policy
------------
* src/features.py and src/modeling.py are reused VERBATIM. features.build_features
  already takes a `horizon` argument and shifts the target, smart-persistence
  reference, solar geometry, clear-sky GHI, calendar and dust-slice column
  correctly for any horizon. Lagged predictors are shifts >= 0 (info available at
  t) at every horizon -> no leakage.
* The CV protocol (12 folds, validation slice, seeds) and the model/metric logic
  are reused from src/train_dust_aware_cv.py via import (that module is NOT
  modified, so step-3 outputs remain byte-identical).

Only the horizon changes: 4, 12, 24 steps = 1 h, 3 h, 6 h.

Run:
    .venv\\Scripts\\python.exe src\\horizon_sweep.py
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
# Reuse the exact CV pieces from step 3 (module import does NOT run its main()).
from train_dust_aware_cv import (
    ANNUAL_HIGH_DUST_THR, MODELS, VARIANTS, _fit_predict, _fold_valid_split,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

HORIZONS = [4, 12, 24]          # steps
HZ_HOURS = {4: 1, 12: 3, 24: 6}  # step -> hours label

# Step-3 pooled baseline numbers (horizon = 4) for the reproduction check.
STEP3_BASELINE = {
    ("LightGBM", "RMSE"): 44.585, ("LightGBM", "skill"): 0.229,
    ("XGBoost", "RMSE"): 44.502, ("XGBoost", "skill"): 0.230,
}
STEP3_N_DAYTIME = 17618
STEP3_N_HIGHDUST = 1933


def run_cv(feat: pd.DataFrame):
    """Run the 12-fold leave-one-month-out CV for the given design frame.
    Returns (pooled_table, improvement_table, n_daytime, n_highdust)."""
    dd = feat[feat["daytime"]].copy()
    pred = {(v, m): pd.Series(index=dd.index, dtype=float)
            for v in VARIANTS for m in MODELS}

    for test_month in range(1, 13):
        test_day = feat[(feat.index.month == test_month) & feat["daytime"]]
        train_all = feat[feat.index.month != test_month]
        train_fit, valid = _fold_valid_split(train_all)
        for variant, cols in VARIANTS.items():
            for model_name in MODELS:
                p = _fit_predict(model_name, cols, train_fit, valid, test_day)
                pred[(variant, model_name)].loc[test_day.index] = p

    y = dd["y"].to_numpy()
    smart = np.clip(dd["pred_smart_persist"].to_numpy(), 0, None)
    dust_target = dd[F.DUST_SLICE_COL].to_numpy()
    hd = dust_target > ANNUAL_HIGH_DUST_THR
    slices = {"all-sky": np.ones(len(dd), bool), "high-dust": hd}
    mean_ghi = {s: float(y[m].mean()) for s, m in slices.items()}
    rmse_ref = {s: M.rmse(y[m], smart[m]) for s, m in slices.items()}

    rows = []

    def _emit(model, variant, p):
        for s, m in slices.items():
            rows.append({"model": model, "variant": variant, "slice": s,
                         **M.metrics(y[m], p[m], mean_ghi[s], rmse_ref[s])})

    _emit("SmartPersistence", "-", smart)
    for variant in VARIANTS:
        for model_name in MODELS:
            _emit(model_name, variant, pred[(variant, model_name)].to_numpy())

    table = pd.DataFrame(rows).set_index(["model", "variant", "slice"])
    table = table[["RMSE", "MAE", "MBE", "nRMSE", "R2", "skill_vs_smart_persist"]]

    imp_rows = []
    for model_name in MODELS:
        for s in slices:
            b = table.loc[(model_name, "baseline", s), "RMSE"]
            d = table.loc[(model_name, "dust_aware", s), "RMSE"]
            imp_rows.append({"model": model_name, "slice": s,
                             "RMSE_baseline": b, "RMSE_dust_aware": d,
                             "RMSE_improve_pct": 100.0 * (b - d) / b})
    improve = pd.DataFrame(imp_rows).set_index(["model", "slice"])
    return table, improve, len(dd), int(hd.sum())


def main() -> int:
    df = F.load_merged()

    master_rows = []       # horizon x model x variant
    improve_rows = []      # horizon x model
    baseline_skill = []    # horizon x model (all-sky)
    repro_ok = True

    for h in HORIZONS:
        hours = HZ_HOURS[h]
        print(f"\n=== HORIZON {hours} h ({h} steps): running 12-fold CV x 2 models x 2 variants ===")
        feat = F.build_features(df, horizon=h)
        table, improve, n_day, n_hd = run_cv(feat)
        print(f"  pooled daytime rows={n_day} | high-dust rows={n_hd}")

        # Reproduction check at 1 h vs step 3.
        if h == 4:
            print("  [reproduction check vs step 3]")
            for model in MODELS:
                got_rmse = table.loc[(model, "baseline", "all-sky"), "RMSE"]
                got_skill = table.loc[(model, "baseline", "all-sky"), "skill_vs_smart_persist"]
                exp_rmse = STEP3_BASELINE[(model, "RMSE")]
                exp_skill = STEP3_BASELINE[(model, "skill")]
                ok = abs(got_rmse - exp_rmse) < 0.02 and abs(got_skill - exp_skill) < 0.002
                repro_ok = repro_ok and ok
                print(f"    {model}: RMSE {got_rmse:.3f} (exp {exp_rmse}) | "
                      f"skill {got_skill:.3f} (exp {exp_skill}) -> {'PASS' if ok else 'FAIL'}")
            repro_ok = repro_ok and (n_day == STEP3_N_DAYTIME) and (n_hd == STEP3_N_HIGHDUST)
            print(f"    rows: daytime {n_day} (exp {STEP3_N_DAYTIME}), "
                  f"high-dust {n_hd} (exp {STEP3_N_HIGHDUST})")
            if not repro_ok:
                print("\nREPRODUCTION CHECK FAILED at 1 h - something regressed. Stopping.")
                return 1

        for model in MODELS:
            for variant in ["baseline", "dust_aware"]:
                master_rows.append({
                    "horizon_h": hours, "model": model, "variant": variant,
                    "allsky_RMSE": table.loc[(model, variant, "all-sky"), "RMSE"],
                    "allsky_nRMSE": table.loc[(model, variant, "all-sky"), "nRMSE"],
                    "allsky_skill": table.loc[(model, variant, "all-sky"), "skill_vs_smart_persist"],
                    "highdust_RMSE": table.loc[(model, variant, "high-dust"), "RMSE"],
                })
            improve_rows.append({
                "horizon_h": hours, "model": model,
                "allsky_improve_pct": improve.loc[(model, "all-sky"), "RMSE_improve_pct"],
                "highdust_improve_pct": improve.loc[(model, "high-dust"), "RMSE_improve_pct"],
            })
            baseline_skill.append({
                "horizon_h": hours, "model": model,
                "baseline_allsky_skill": table.loc[(model, "baseline", "all-sky"), "skill_vs_smart_persist"],
                "baseline_allsky_RMSE": table.loc[(model, "baseline", "all-sky"), "RMSE"],
            })

    master = pd.DataFrame(master_rows).set_index(["horizon_h", "model", "variant"])
    improve = pd.DataFrame(improve_rows).set_index(["horizon_h", "model"])
    skill_df = pd.DataFrame(baseline_skill).set_index(["horizon_h", "model"])

    master.to_csv(RESULTS / "horizon_sweep_metrics.csv", float_format="%.4f")
    improve.to_csv(RESULTS / "horizon_improvement.csv", float_format="%.4f")

    _plot_dust_benefit(improve)

    # ---- Summary ----------------------------------------------------------- #
    print("\n" + "=" * 82)
    print("HORIZON SWEEP SUMMARY (month-blocked CV; dust-aware vs baseline)")
    print("=" * 82)
    print("MASTER TABLE (pooled):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}",
                           "display.width", 200):
        print(master.to_string())
    print("-" * 82)
    print("RMSE IMPROVEMENT FROM DUST (%, + = dust helps):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}",
                           "display.width", 200):
        print(improve.to_string())
    print("-" * 82)
    print("BASELINE SKILL vs PERSISTENCE across horizons (context):")
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}",
                           "display.width", 200):
        print(skill_df.to_string())
    print("-" * 82)
    print("Saved: results/horizon_sweep_metrics.csv, results/horizon_improvement.csv")
    print("Figure: figures/horizon_dust_benefit.png")
    return 0


def _plot_dust_benefit(improve: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    horizons = sorted({h for h, _ in improve.index})
    for ax, model in zip(axes, MODELS):
        allsky = [improve.loc[(h, model), "allsky_improve_pct"] for h in horizons]
        highd = [improve.loc[(h, model), "highdust_improve_pct"] for h in horizons]
        ax.plot(horizons, allsky, marker="o", color="#64748b", label="all-sky")
        ax.plot(horizons, highd, marker="s", color="#b45309", label="high-dust (>0.4413)")
        for xh, ya, yh in zip(horizons, allsky, highd):
            ax.annotate(f"{ya:+.1f}%", (xh, ya), fontsize=8, ha="center", va="bottom")
            ax.annotate(f"{yh:+.1f}%", (xh, yh), fontsize=8, ha="center", va="top")
        ax.axhline(0, color="#334155", lw=1)
        ax.set_title(model)
        ax.set_xlabel("forecast horizon [h]")
        ax.set_xticks(horizons)
    axes[0].set_ylabel("RMSE improvement from dust [%]  (+ = dust helps)")
    axes[0].legend()
    fig.suptitle("Does dust help more at longer horizons? (month-blocked CV)")
    fig.tight_layout()
    fig.savefig(FIGURES / "horizon_dust_benefit.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
