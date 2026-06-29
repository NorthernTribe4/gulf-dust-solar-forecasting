"""
Phase 2a data-acquisition gate for the Gulf dust-solar forecasting project.

Goal: prove we can obtain aligned, granular solar-irradiance (NSRDB) and
dust-aerosol (CAMS EAC4) data for one Gulf site (Doha, Qatar), then report
exactly what we got. This script performs NO modelling.

Run:
    .venv\\Scripts\\python.exe src\\acquire_data.py
"""

from __future__ import annotations

import io
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SITE_NAME = "Doha, Qatar"
LAT = 25.2854
LON = 51.5310
WKT = f"POINT({LON} {LAT})"  # NSRDB expects POINT(longitude latitude)

# CAMS bounding box around Doha: [North, West, South, East]
CAMS_AREA = [26.0, 50.5, 24.5, 52.5]

# NREL developer domain (moved to developer.nlr.gov; old developer.nrel.gov
# retired May 2026).
NREL_BASE = "https://developer.nlr.gov"
NSRDB_QUERY_URL = f"{NREL_BASE}/api/solar/nsrdb_data_query.json"

# NSRDB attributes we want in the download.
NSRDB_ATTRIBUTES = [
    "ghi",
    "dni",
    "dhi",
    "solar_zenith_angle",
    "air_temperature",
    "aod",
]

# Project paths (resolved relative to repo root = parent of this file's dir).
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (RAW, PROCESSED, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Report buffer: everything appended here is printed AND saved.
_REPORT: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _REPORT.append(msg)


def section(title: str) -> None:
    bar = "=" * 70
    log("")
    log(bar)
    log(title)
    log(bar)


# --------------------------------------------------------------------------- #
# Step 1: NSRDB availability query (the core gate check)
# --------------------------------------------------------------------------- #
def nsrdb_query(api_key: str) -> dict:
    section("STEP 1 - NSRDB AVAILABILITY QUERY")
    params = {"api_key": api_key, "wkt": WKT}
    log(f"GET {NSRDB_QUERY_URL}")
    log(f"     wkt={WKT}")
    resp = requests.get(NSRDB_QUERY_URL, params=params, timeout=60)
    log(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()

    outputs = data.get("outputs", [])
    log(f"Datasets returned for {SITE_NAME}: {len(outputs)}")
    for o in outputs:
        log("")
        log(f"  name           : {o.get('name')}")
        log(f"  displayName    : {o.get('displayName')}")
        log(f"  availableYears : {o.get('availableYears')}")
        log(f"  availableIntervals (min): {o.get('availableIntervals')}")
    return data


def choose_dataset(query: dict) -> tuple[dict, int, int, str]:
    """Pick dataset with the most recent full year at the finest interval.

    Returns (output_dict, year, interval_minutes, download_link_template).
    """
    outputs = query.get("outputs", [])
    if not outputs:
        raise RuntimeError("NSRDB query returned no datasets for this location.")

    best = None  # (year, -interval, output, interval, link)
    for o in outputs:
        # Skip typical-year products (TMY/TGY/TDY) whose "years" are strings
        # like 'tmy', 'tdy-2022'; we want a real chronological year.
        years = []
        for y in (o.get("availableYears") or []):
            try:
                years.append(int(y))
            except (TypeError, ValueError):
                continue
        intervals = [int(i) for i in (o.get("availableIntervals") or [])]
        links = o.get("links") or []
        if not years or not intervals:
            continue
        year = max(years)
        interval = min(intervals)  # finest interval
        # Find a link matching this year+interval if available.
        link = None
        for lk in links:
            if int(lk.get("year", -1)) == year and int(lk.get("interval", -1)) == interval:
                link = lk.get("link")
                break
        if link is None and links:
            link = links[0].get("link")
        cand = (year, -interval, o, interval, link)
        if best is None or cand[:2] > best[:2]:
            best = cand

    if best is None:
        raise RuntimeError("No dataset exposed usable years/intervals.")
    year, _, output, interval, link = best
    section("STEP 1b - SELECTED PRODUCT")
    log(f"  dataset  : {output.get('name')} ({output.get('displayName')})")
    log(f"  year     : {year}")
    log(f"  interval : {interval} min")
    log(f"  link tmpl: {link}")
    return output, year, interval, link


# --------------------------------------------------------------------------- #
# Step 2: NSRDB download (small slice)
# --------------------------------------------------------------------------- #
def nsrdb_download(api_key: str, year: int, interval: int, link: str | None) -> pd.DataFrame:
    section("STEP 2 - NSRDB DOWNLOAD")

    # Build the download URL. Prefer the link template the query handed back so
    # we stay on whatever endpoint/domain the API itself advertises.
    if link:
        # Strip any existing query string; we supply our own params.
        base_url = link.split("?")[0]
    else:
        base_url = f"{NREL_BASE}/api/nsrdb/v2/solar/psm3-download.csv"

    params = {
        "api_key": api_key,
        "wkt": WKT,
        "names": str(year),
        "interval": str(interval),
        "utc": "true",
        "leap_day": "false",
        "attributes": ",".join(NSRDB_ATTRIBUTES),
        "email": "aarush0409@gmail.com",
        "full_name": "Aarush",
        "affiliation": "independent-research",
        "reason": "research",
        "mailing_list": "false",
    }
    log(f"GET {base_url}")
    log(f"     names={year} interval={interval} utc=true attributes={','.join(NSRDB_ATTRIBUTES)}")

    # The endpoint intermittently returns 400 "Data processing failure"; retry.
    import time

    resp = None
    for attempt in range(1, 5):
        resp = requests.get(base_url, params=params, timeout=300)
        log(f"  attempt {attempt}: HTTP {resp.status_code}")
        if resp.status_code == 200:
            break
        log(f"    body: {resp.text[:300]}")
        time.sleep(5)
    if resp is None or resp.status_code != 200:
        resp.raise_for_status()

    raw_text = resp.text
    raw_path = RAW / f"nsrdb_doha_{year}_{interval}min.csv"
    raw_path.write_text(raw_text, encoding="utf-8")
    log(f"Saved raw NSRDB -> {raw_path} ({len(raw_text)} bytes)")

    # NSRDB CSV: row 0 = metadata field names, row 1 = metadata values,
    # row 2 = data column header, rows 3+ = data.
    meta = pd.read_csv(io.StringIO(raw_text), nrows=1)
    log("NSRDB metadata header:")
    log(textwrap.indent(meta.to_string(index=False), "    "))

    df = pd.read_csv(io.StringIO(raw_text), skiprows=2)
    # Build UTC timestamp from Year/Month/Day/Hour/Minute columns.
    tcols = ["Year", "Month", "Day", "Hour", "Minute"]
    missing = [c for c in tcols if c not in df.columns]
    if missing:
        raise RuntimeError(f"NSRDB CSV missing time columns: {missing}; got {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df[tcols].rename(columns=str.lower), utc=True)
    df = df.set_index("timestamp").sort_index()

    clean_path = PROCESSED / f"nsrdb_doha_{year}_clean.csv"
    df.to_csv(clean_path)
    log(f"Parsed NSRDB rows: {len(df)}  columns: {list(df.columns)}")
    log(f"Saved clean NSRDB -> {clean_path}")
    return df


# --------------------------------------------------------------------------- #
# Step 3: CAMS EAC4 download (same year)
# --------------------------------------------------------------------------- #
def cams_download(year: int) -> Path:
    section("STEP 3 - CAMS EAC4 DOWNLOAD")
    import cdsapi

    target = RAW / f"cams_eac4_doha_{year}.nc"
    if target.exists() and target.stat().st_size > 0:
        log(f"CAMS file already present, skipping download -> {target}")
        return target

    dataset = "cams-global-reanalysis-eac4"
    variables = [
        "total_aerosol_optical_depth_550nm",
        "dust_aerosol_optical_depth_550nm",
    ]
    request = {
        "variable": variables,
        "date": f"{year}-01-01/{year}-12-31",
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
        "area": CAMS_AREA,  # [N, W, S, E]
        "data_format": "netcdf",
    }
    log(f"dataset : {dataset}")
    log(f"variables: {variables}")
    log(f"area [N,W,S,E]: {CAMS_AREA}")
    log(f"date    : {year}-01-01/{year}-12-31, 3-hourly")

    client = cdsapi.Client()
    try:
        client.retrieve(dataset, request, str(target))
    except Exception as exc:  # noqa: BLE001 - report whatever the API says
        log(f"CAMS retrieve failed with data_format=netcdf: {exc}")
        log("Retrying with legacy 'format' key...")
        request.pop("data_format", None)
        request["format"] = "netcdf"
        client.retrieve(dataset, request, str(target))

    log(f"Saved raw CAMS -> {target} ({target.stat().st_size} bytes)")
    return target


def load_cams(nc_path: Path) -> pd.DataFrame:
    """Load CAMS NetCDF, pick grid cell nearest Doha, return UTC-indexed df."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    log("CAMS dataset variables: " + ", ".join(ds.data_vars))
    log("CAMS coords: " + ", ".join(ds.coords))

    # Coordinate names differ across CDS backends.
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "valid_time" if "valid_time" in ds.coords else "time"

    cell = ds.sel({lat_name: LAT, lon_name: LON}, method="nearest")
    sel_lat = float(cell[lat_name].values)
    sel_lon = float(cell[lon_name].values)
    log(f"Nearest CAMS cell to Doha: lat={sel_lat:.3f} lon={sel_lon:.3f}")

    # Identify total and dust AOD variables (EAC4 short names: aod550, duaod550).
    total_name = next((v for v in ("aod550", "total_aod550") if v in cell.data_vars), None)
    dust_name = next((v for v in ("duaod550", "dust_aod550") if v in cell.data_vars), None)
    if total_name is None or dust_name is None:
        # Fall back to anything containing the right token.
        for v in cell.data_vars:
            lv = v.lower()
            if total_name is None and "aod" in lv and "du" not in lv and "dust" not in lv:
                total_name = v
            if dust_name is None and ("duaod" in lv or "dust" in lv):
                dust_name = v
    log(f"CAMS total AOD var: {total_name}; dust AOD var: {dust_name}")

    out = pd.DataFrame(
        {
            "cams_total_aod550": cell[total_name].values,
            "cams_dust_aod550": cell[dust_name].values,
        },
        index=pd.to_datetime(cell[time_name].values, utc=True),
    )
    out.index.name = "timestamp"
    out = out.sort_index()
    ds.close()
    return out


# --------------------------------------------------------------------------- #
# Step 4: align and merge
# --------------------------------------------------------------------------- #
def align_merge(nsrdb: pd.DataFrame, cams: pd.DataFrame, year: int) -> pd.DataFrame:
    section("STEP 4 - ALIGN AND MERGE")

    # Normalise NSRDB column names we care about to stable keys.
    rename = {
        "GHI": "ghi",
        "DNI": "dni",
        "DHI": "dhi",
        "Solar Zenith Angle": "solar_zenith_angle",
        "Temperature": "air_temperature",
        "Aerosol Optical Depth": "nsrdb_aod",
        "AOD": "nsrdb_aod",
    }
    nsrdb = nsrdb.rename(columns={k: v for k, v in rename.items() if k in nsrdb.columns})

    keep = [c for c in ["ghi", "dni", "dhi", "solar_zenith_angle", "air_temperature", "nsrdb_aod"] if c in nsrdb.columns]
    base = nsrdb[keep].copy()

    # Determine NSRDB native step (minutes) so the forward-fill carries each
    # 3-hourly CAMS value across exactly one CAMS block of NSRDB timestamps.
    step_min = int(round(base.index.to_series().diff().dropna().median().total_seconds() / 60))
    block_steps = max(1, round(180 / step_min))  # NSRDB steps per 3h CAMS block
    ffill_limit = block_steps - 1
    log(f"NSRDB index tz: {base.index.tz}; CAMS index tz: {cams.index.tz}")
    log(f"NSRDB native step: {step_min} min; CAMS block = {block_steps} NSRDB steps")
    log(f"Alignment method: reindex CAMS (3-hourly) onto NSRDB ({step_min}-min) index, "
        f"forward-fill within each 3-hour block (ffill limit={ffill_limit} steps).")

    cams_on_base = cams.reindex(base.index.union(cams.index)).sort_index()
    cams_on_base = cams_on_base.ffill(limit=ffill_limit)
    cams_on_base = cams_on_base.reindex(base.index)

    merged = base.join(cams_on_base, how="left")
    merged_path = PROCESSED / "doha_merged.csv"
    merged.to_csv(merged_path)
    log(f"Merged rows: {len(merged)}  columns: {list(merged.columns)}")
    log(f"Saved merged -> {merged_path}")
    return merged


# --------------------------------------------------------------------------- #
# Step 5: verification report + plots
# --------------------------------------------------------------------------- #
def _range_str(df: pd.DataFrame) -> str:
    return f"{df.index.min()} -> {df.index.max()}"


def verify(nsrdb: pd.DataFrame, cams: pd.DataFrame, merged: pd.DataFrame) -> None:
    section("STEP 5 - VERIFICATION REPORT")

    def _res_min(df: pd.DataFrame) -> int:
        return int(round(df.index.to_series().diff().dropna().median().total_seconds() / 60))

    nsrdb_res = _res_min(nsrdb)
    cams_res = _res_min(cams)
    merge_res = _res_min(merged)
    log("[Date range / row count / native resolution]")
    log(f"  NSRDB : {_range_str(nsrdb)} | rows={len(nsrdb)} | native={nsrdb_res} min")
    log(f"  CAMS  : {_range_str(cams)} | rows={len(cams)} | native={cams_res} min")
    log(f"  MERGE : {_range_str(merged)} | rows={len(merged)} | grid={merge_res} min (NSRDB native)")

    log("")
    log("[Percent missing per key column - merged table]")
    for col in merged.columns:
        pct = 100.0 * merged[col].isna().mean()
        log(f"  {col:22s}: {pct:6.2f}%")

    # GHI stats
    log("")
    if "ghi" in merged.columns:
        ghi = merged["ghi"]
        daytime = merged[ghi > 0]
        log("[GHI]")
        log(f"  mean={ghi.mean():.2f}  min={ghi.min():.2f}  max={ghi.max():.2f}")
        log(f"  daytime rows (GHI>0): {len(daytime)} of {len(merged)} "
            f"({100.0*len(daytime)/len(merged):.1f}%)")
    else:
        log("[GHI] column not present - cannot compute.")

    # Dust AOD stats
    log("")
    if "cams_dust_aod550" in merged.columns:
        dust = merged["cams_dust_aod550"].dropna()
        log("[Dust AOD 550nm]")
        log(f"  mean={dust.mean():.4f}  median={dust.median():.4f}  max={dust.max():.4f}")
        if len(dust):
            p90 = dust.quantile(0.90)
            high_mask = merged["cams_dust_aod550"] > p90
            n_high = int(high_mask.sum())
            log(f"  high-dust threshold (90th pct): {p90:.4f}")
            log(f"  high-dust timesteps: {n_high} "
                f"({100.0*n_high/len(merged):.1f}% of all rows)")
            if "ghi" in merged.columns:
                n_high_day = int((high_mask & (merged["ghi"] > 0)).sum())
                log(f"  high-dust AND daytime (GHI>0): {n_high_day} "
                    f"({100.0*n_high_day/max(n_high,1):.1f}% of high-dust rows)")
    else:
        log("[Dust AOD] column not present - cannot compute.")

    _plots(merged)


def _plots(merged: pd.DataFrame) -> None:
    log("")
    log("[Diagnostic plots]")

    # (a) GHI over time
    if "ghi" in merged.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(merged.index, merged["ghi"], lw=0.4, color="#d97706")
        ax.set_title(f"GHI over time - {SITE_NAME}")
        ax.set_ylabel("GHI (W/m^2)")
        ax.set_xlabel("UTC")
        fig.tight_layout()
        p = FIGURES / "ghi_timeseries.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        log(f"  saved {p}")

    # (b) dust AOD over time
    if "cams_dust_aod550" in merged.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(merged.index, merged["cams_dust_aod550"], lw=0.5, color="#b45309")
        ax.set_title(f"Dust AOD 550nm over time - {SITE_NAME}")
        ax.set_ylabel("Dust AOD 550nm")
        ax.set_xlabel("UTC")
        fig.tight_layout()
        p = FIGURES / "dust_aod_timeseries.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        log(f"  saved {p}")

    # (c) scatter daytime GHI vs dust AOD
    if "ghi" in merged.columns and "cams_dust_aod550" in merged.columns:
        day = merged[(merged["ghi"] > 0) & merged["cams_dust_aod550"].notna()]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(day["cams_dust_aod550"], day["ghi"], s=4, alpha=0.3, color="#7c2d12")
        ax.set_title("Daytime GHI vs Dust AOD 550nm")
        ax.set_xlabel("Dust AOD 550nm")
        ax.set_ylabel("GHI (W/m^2)")
        fig.tight_layout()
        p = FIGURES / "ghi_vs_dust_scatter.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        log(f"  saved {p}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("NREL_API_KEY")
    if not api_key:
        print("ERROR: NREL_API_KEY not found in .env", file=sys.stderr)
        return 2

    section(f"DATA GATE - {SITE_NAME}  (lat={LAT}, lon={LON})")
    log(f"NSRDB WKT: {WKT}")

    query = nsrdb_query(api_key)
    output, year, interval, link = choose_dataset(query)

    nsrdb = nsrdb_download(api_key, year, interval, link)

    nc_path = cams_download(year)
    cams = load_cams(nc_path)

    merged = align_merge(nsrdb, cams, year)
    verify(nsrdb, cams, merged)

    # Persist the report.
    report_path = RESULTS / "data_gate_report.txt"
    report_path.write_text("\n".join(_REPORT), encoding="utf-8")

    section("SAVED FILES")
    log(f"  raw NSRDB    : data/raw/nsrdb_doha_{year}_{interval}min.csv")
    log(f"  clean NSRDB  : data/processed/nsrdb_doha_{year}_clean.csv")
    log(f"  raw CAMS     : {nc_path.relative_to(ROOT)}")
    log(f"  merged       : data/processed/doha_merged.csv")
    log(f"  report       : results/data_gate_report.txt")
    log(f"  figures      : figures/ghi_timeseries.png, dust_aod_timeseries.png, ghi_vs_dust_scatter.png")

    # Rewrite report to include the saved-files section too.
    report_path.write_text("\n".join(_REPORT), encoding="utf-8")
    log("")
    log("GATE COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
