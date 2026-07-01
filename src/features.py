"""
Feature engineering for the Gulf dust-solar GHI forecaster.

This module is shared by the dust-BLIND baseline (Phase 3 step 1) and the later
dust-aware model, so the split logic and the non-aerosol feature construction
live here and are reused verbatim.

FORECAST TASK
-------------
* Target : GHI 1 hour ahead. At 15-min resolution that is HORIZON_STEPS = 4
           steps ahead:  y(t) = GHI(t + 1h).
* We predict GHI(t+1h) using only information available at issue time t, EXCEPT
  for quantities that are deterministic astronomy (solar geometry, clear-sky
  GHI) and calendar values at the target time t+1h -- those are genuinely known
  in advance and are standard, non-leaking inputs in solar forecasting. They are
  also exactly what a smart-persistence reference needs.

LEAKAGE RULES
-------------
* Past/known-at-t predictors (GHI, clear-sky index, temperature) are taken at
  shifts >= 0 (i.e. t, t-15min, ...). They never peek past t.
* Deterministic target-time predictors (solar zenith, clear-sky GHI, calendar)
  are taken at t+1h -- known without observing the future.
* dni / dhi are NEVER used (they leak the target).
* Aerosol columns (nsrdb_aod, cams_total_aod550, cams_dust_aod550) are NOT
  touched here at all -- the dust-aware step adds them on top. This keeps the
  baseline strictly dust-blind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

# --------------------------------------------------------------------------- #
# Site + task constants
# --------------------------------------------------------------------------- #
SITE_LAT = 25.2854
SITE_LON = 51.5310
SITE_ALT = 10.0  # m, from NSRDB metadata for the Doha cell

STEP_MIN = 15
HORIZON_STEPS = 4          # 4 x 15 min = 1 hour ahead
DAY_STEPS = 96             # 24 h = 96 steps (diurnal anchor lag)
GHI_LAGS = [0, 1, 2, 3, 4, DAY_STEPS]   # steps back from t (0 = current value)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED = ROOT / "data" / "processed" / "doha_merged.csv"

# Columns that must never be inputs to the baseline.
LEAKY_COLS = ["ghi", "dni", "dhi"]                      # target + target-leakers
AEROSOL_COLS = ["nsrdb_aod", "cams_total_aod550", "cams_dust_aod550"]  # next step


def load_merged(path: Path | str = DEFAULT_MERGED) -> pd.DataFrame:
    """Load the merged data-gate CSV with a UTC DatetimeIndex at 15-min steps."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # Enforce a regular 15-min grid (the data gate already guarantees this, but
    # asserting it makes the shift-based lags safe to interpret as time lags).
    step = df.index.to_series().diff().dropna().median()
    assert step == pd.Timedelta(minutes=STEP_MIN), f"unexpected step {step}"
    return df


def add_clearsky(df: pd.DataFrame) -> pd.DataFrame:
    """Add clear-sky GHI (Ineichen, climatological Linke turbidity) and the
    clear-sky index kt = GHI / GHI_cs. kt is dust-blind: the Ineichen model uses
    a turbidity climatology, not our aerosol columns."""
    loc = pvlib.location.Location(SITE_LAT, SITE_LON, tz="UTC", altitude=SITE_ALT)
    cs = loc.get_clearsky(df.index, model="ineichen")
    out = df.copy()
    out["ghi_cs"] = cs["ghi"].to_numpy()
    # Clear-sky index only where the sun is meaningfully up; else 0 (night).
    eps = 1.0  # W/m^2 floor to avoid dividing by tiny clear-sky values at dusk
    kt = np.where(out["ghi_cs"].to_numpy() > eps,
                  out["ghi"].to_numpy() / np.maximum(out["ghi_cs"].to_numpy(), eps),
                  0.0)
    # Clip kt to a sane physical band (thin cloud/edge effects can exceed 1).
    out["kt"] = np.clip(kt, 0.0, 1.5)
    return out


def build_features(df: pd.DataFrame, horizon: int = HORIZON_STEPS) -> pd.DataFrame:
    """Return a design frame indexed by issue time t.

    Contains:
      * baseline (dust-blind) feature columns  -> see BASE_FEATURES below
      * 'y'                : GHI at t+horizon (target)
      * 'daytime'          : sun up at the TARGET time (solar_zenith < 90)
      * 'ghi_cs_target'    : clear-sky GHI at t+horizon (helper)
      * 'pred_smart_persist', 'pred_plain_persist' : reference forecasts
    Rows with any NaN in features/target are dropped (lag/lead warm-up).
    """
    d = add_clearsky(df)
    f = pd.DataFrame(index=d.index)

    # --- Past predictors (known at t): lagged GHI and clear-sky index -------- #
    for L in GHI_LAGS:
        f[f"ghi_lag{L}"] = d["ghi"].shift(L)
        f[f"kt_lag{L}"] = d["kt"].shift(L)
    # Short-window recent dynamics (last hour), known at t.
    f["ghi_roll4_mean"] = d["ghi"].rolling(4).mean()
    f["ghi_roll4_std"] = d["ghi"].rolling(4).std()
    f["kt_roll4_mean"] = d["kt"].rolling(4).mean()
    # Air temperature is only known up to t -> use the value at t (and 1h ago).
    f["air_temperature_lag0"] = d["air_temperature"].shift(0)
    f["air_temperature_lag4"] = d["air_temperature"].shift(horizon)

    # --- Deterministic target-time predictors (known in advance) ------------ #
    # Solar geometry and clear-sky GHI at t+horizon are pure astronomy.
    f["solar_zenith_target"] = d["solar_zenith_angle"].shift(-horizon)
    f["cos_zenith_target"] = np.cos(np.radians(f["solar_zenith_target"].clip(0, 90)))
    f["ghi_cs_target"] = d["ghi_cs"].shift(-horizon)

    # Calendar features of the TARGET time (index shifted forward by horizon).
    target_time = d.index + pd.Timedelta(minutes=STEP_MIN * horizon)
    hour = target_time.hour + target_time.minute / 60.0
    doy = target_time.dayofyear
    f["hour_target"] = hour
    f["doy_target"] = doy
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # --- Target ------------------------------------------------------------- #
    f["y"] = d["ghi"].shift(-horizon)

    # --- Daytime mask (evaluate only when sun is up at target time) --------- #
    f["daytime"] = f["solar_zenith_target"] < 90.0

    # --- Reference forecasts ------------------------------------------------ #
    # Smart persistence: hold the clear-sky index from t, project onto the known
    # clear-sky GHI at t+horizon.  pred = kt(t) * GHI_cs(t+horizon).
    f["pred_smart_persist"] = (d["kt"].shift(0).to_numpy()) * f["ghi_cs_target"].to_numpy()
    # Plain persistence: GHI(t+horizon) ~= GHI(t).
    f["pred_plain_persist"] = d["ghi"].shift(0)

    f = f.dropna()
    return f


# Baseline (dust-blind) feature columns = everything in the design frame that is
# not the target, a helper, or a reference forecast.
_NON_FEATURE = {
    "y", "daytime", "ghi_cs_target",
    "pred_smart_persist", "pred_plain_persist",
}


def base_feature_columns(feat: pd.DataFrame) -> list[str]:
    """Dust-blind feature column list (order stable). ghi_cs_target is kept as a
    predictor -- it is deterministic astronomy -- but its name is in the helper
    set, so we re-add it explicitly here."""
    cols = [c for c in feat.columns if c not in _NON_FEATURE]
    cols.append("ghi_cs_target")  # deterministic, legitimate predictor
    return cols


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
# Train Jan-Sep 2022, hold out Oct-Dec 2022. A validation slice (September) is
# carved from the tail of train for early stopping.
TRAIN_END = pd.Timestamp("2022-10-01 00:00", tz="UTC")   # test starts here
VALID_START = pd.Timestamp("2022-09-01 00:00", tz="UTC")  # valid = Sep (in train)


def chrono_split(feat: pd.DataFrame):
    """Return (train_fit, valid, test) frames split by issue time t.

    * test      : t >= TRAIN_END               (Oct-Dec)
    * valid     : VALID_START <= t < TRAIN_END  (September, for early stopping)
    * train_fit : t < VALID_START               (Jan-Aug)
    'train_full' (Jan-Sep) = train_fit + valid is what final metrics-fair models
    could use; here we fit on train_fit and early-stop on valid.
    """
    test = feat[feat.index >= TRAIN_END]
    train_full = feat[feat.index < TRAIN_END]
    valid = train_full[train_full.index >= VALID_START]
    train_fit = train_full[train_full.index < VALID_START]
    return train_fit, valid, test, train_full
