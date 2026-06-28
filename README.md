# Gulf Dust–Solar Forecasting

Research code for a paper targeted at **BDA 2026** (Springer LNCS proceedings).

## Research Question

> Does an explicit dust/aerosol-attenuation feature improve short-horizon solar
> irradiance forecasting in arid Gulf conditions, where standard forecasts
> underperform under high atmospheric dust loading?

Arid Gulf sites experience frequent, high-magnitude dust events that scatter and
absorb incoming shortwave radiation. Standard irradiance forecasts that do not
account for aerosol loading tend to degrade sharply during these events. This
project tests whether adding an explicit aerosol-attenuation feature (derived
from aerosol optical depth) to short-horizon forecasting models measurably
improves accuracy relative to baselines without such a feature.

## Datasets

> **Note:** No data is downloaded in this scaffolding stage. The entries below
> document the planned public sources and how they will be accessed.

### NSRDB — solar irradiance (target + predictors)
- **Source:** NREL National Solar Radiation Database.
- **Variables:** GHI, DNI, DHI (global/direct/diffuse horizontal irradiance) for
  Gulf sites.
- **Access:** NREL developer API. Requires a free API key
  ([request one here](https://developer.nrel.gov/signup/)), stored as
  `NREL_API_KEY` in a local `.env` file.

### CAMS — aerosol optical depth (the candidate feature)
- **Source:** Copernicus Atmosphere Monitoring Service.
- **Variable:** aerosol optical depth (AOD) at 550 nm.
- **Access:** Atmosphere Data Store (ADS) via the `cdsapi` Python client.
  Requires free registration ([ADS](https://ads.atmosphere.copernicus.eu/)) and a
  `~/.cdsapirc` credentials file (see Setup below).

## Folder Structure

```
gulf-dust-solar-forecasting/
├── data/
│   ├── raw/            # Raw downloaded data (git-ignored, never committed)
│   └── processed/      # Cleaned, merged, feature-engineered datasets
├── src/                # Source code (data ingestion, features, models, eval)
├── notebooks/          # Exploratory analysis and experiment notebooks
├── results/            # Metrics, model outputs, experiment artifacts
├── figures/            # Plots and figures for the paper
├── paper/              # LaTeX manuscript (LNCS) and paper assets
├── requirements.txt
├── .env.example        # Template for NREL_API_KEY
├── .gitignore
└── README.md
```

## Setup (Windows / PowerShell)

Exact commands to recreate the environment from a fresh clone:

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
# Create %USERPROFILE%\.cdsapirc with the following two lines:
#   url: https://ads.atmosphere.copernicus.eu/api
#   key: <your-ads-api-key>
```

> If `Activate.ps1` is blocked, allow scripts for the current session:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

## Dependencies

`numpy`, `pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`, `pvlib`,
`cdsapi`, `requests`, `python-dotenv`, `jupyter`.

## Status

Project scaffolding only. No data ingestion or modelling code yet.
