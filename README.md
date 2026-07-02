# Gulf Dust and Solar Forecasting

Research code for a paper targeted at BDA 2026 (Springer LNCS proceedings).

## Research question

> Does an explicit dust/aerosol-attenuation feature improve short-horizon solar
> irradiance forecasting in arid Gulf conditions, where standard forecasts
> underperform under high atmospheric dust loading?

Gulf sites get frequent, strong dust events that scatter and absorb incoming
shortwave radiation. Irradiance forecasts that ignore aerosol loading tend to
degrade during those events. This project tests whether adding a dust aerosol
optical depth feature to short-horizon models measurably improves accuracy over
baselines that do not use it.

The short answer, on 2022 data for Doha, is mostly no. See "Results so far" below.

## Datasets

### NSRDB (solar irradiance: target and predictors)
NREL's National Solar Radiation Database supplies GHI, DNI and DHI (global,
direct and diffuse horizontal irradiance) plus solar geometry and temperature for
the Doha cell at 15-minute resolution. Access is through the NREL developer API,
which needs a free API key ([request one here](https://developer.nrel.gov/signup/))
stored as `NREL_API_KEY` in a local `.env` file.

### CAMS (aerosol optical depth: the candidate feature)
The Copernicus Atmosphere Monitoring Service EAC4 reanalysis supplies total and
dust aerosol optical depth at 550 nm, 3-hourly, over a small box around Doha.
Access is through the Atmosphere Data Store with the `cdsapi` client, which needs
free registration at [ADS](https://ads.atmosphere.copernicus.eu/) and a
`~/.cdsapirc` credentials file (see Setup).

## Folder structure

```
gulf-dust-solar-forecasting/
├── data/
│   ├── raw/            # Raw downloaded data (git-ignored, never committed)
│   └── processed/      # Cleaned, merged, feature-engineered datasets
├── src/                # Data acquisition, features, models, evaluation
├── notebooks/          # Exploratory analysis
├── results/            # Metrics tables and run notes
├── figures/            # Plots for the paper
├── paper/              # LaTeX manuscript (LNCS) and assets
├── requirements.txt
├── .env.example        # Template for NREL_API_KEY
├── .gitignore
└── README.md
```

## Setup (Windows / PowerShell)

Commands to recreate the environment from a fresh clone:

```powershell
# 1. Clone and enter the project
git clone https://github.com/NorthernTribe4/gulf-dust-solar-forecasting.git
cd gulf-dust-solar-forecasting

# 2. Create and activate a Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure the NREL API key
Copy-Item .env.example .env
# Then edit .env and set NREL_API_KEY to your key.

# 5. Configure CAMS / cdsapi credentials
# Create %USERPROFILE%\.cdsapirc with these two lines:
#   url: https://ads.atmosphere.copernicus.eu/api
#   key: <your-ads-api-key>
```

If `Activate.ps1` is blocked, allow scripts for the current session with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

Note on this machine: Windows Smart App Control blocks the native DLLs in recent
scikit-learn/lightgbm/xgboost wheels, and `jupyter` trips the Windows long-path
limit during install. `requirements.txt` pins versions that work around the first
problem; install without `jupyter` to avoid the second. The header of
`requirements.txt` has the details.

## Pipeline

Run these from the repo root with the virtualenv active.

1. `python src/acquire_data.py` downloads and merges NSRDB and CAMS, then writes a
   verification report and diagnostic figures (the data gate).
2. `python src/train_baseline.py` trains the dust-blind 1-hour-ahead GHI baseline.
3. `python src/train_dust_aware.py` adds the dust features and runs the first
   ablation on a single Jan-Sep / Oct-Dec split.
4. `python src/train_dust_aware_cv.py` reruns the ablation under month-blocked
   cross-validation so dust events actually appear in held-out data.
5. `python src/horizon_sweep.py` repeats the CV ablation at 1, 3 and 6 hours.

`src/features.py` holds the feature construction and split logic; `src/modeling.py`
holds the shared model definitions and metrics. Both are reused unchanged across
steps so the baseline-vs-dust comparison stays controlled.

## Results so far

Target is GHI one hour ahead (and 3 and 6 hours in the sweep), evaluated on
daytime timesteps against a smart-persistence reference.

- The dust-blind baseline beats smart persistence (skill about 0.25 at 1 hour).
- Adding CAMS dust AOD does not help at 1 hour. Under month-blocked CV, which does
  put real dust storms in the test folds, it is still slightly negative overall
  and on the true high-dust slice.
- Across horizons the dust feature helps a little on the all-sky set at 6 hours
  (XGBoost about +7%), but not on the high-dust slice at any horizon. So it acts
  more like a general atmospheric-state signal than a dust-event predictor.

The `results/*_notes.txt` files record the numbers and the reasoning behind each
of these conclusions.

## Dependencies

numpy, pandas, scikit-learn, lightgbm, xgboost, matplotlib, pvlib, cdsapi,
requests, python-dotenv. See `requirements.txt` for pinned versions.
