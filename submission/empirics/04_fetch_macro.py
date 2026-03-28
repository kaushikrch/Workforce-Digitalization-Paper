#!/usr/bin/env python3
"""
04_fetch_macro.py — Fetch macro-level control variables from FRED and BLS.

Sources (all free, no authentication required):
  1. FRED (Federal Reserve Economic Data):
     - ECOMSA: E-commerce retail sales (quarterly, seasonally adjusted)
     - RSXFSN: Advance retail sales excluding food services
     - MRTSSM44X72USS: Total retail trade sales
     - CES4200000001: Retail trade employment
     - CES4200000008: Retail trade average hourly earnings
     - CPIAUCSL: CPI (urban consumers, all items)

  2. Census Bureau:
     - Quarterly e-commerce as share of total retail sales

  3. BLS JOLTS:
     - Retail trade turnover, quits, hires (monthly, industry-level)

Output: data/raw/macro_controls.csv
  Columns: year, ecom_share, retail_employment, retail_avg_wage,
           retail_turnover_rate, cpi, retail_sales_total

Usage:
  python 04_fetch_macro.py
"""

import csv
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# FRED API (no key needed for small volume; key recommended for production)
# ---------------------------------------------------------------------------

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs and their descriptions
FRED_SERIES = {
    # E-commerce share data (quarterly)
    "ECOMPCTSA": "E-commerce as % of total retail (quarterly, SA)",
    # Retail employment (monthly)
    "CES4200000001": "Retail trade employment (thousands, SA)",
    # Retail avg hourly earnings (monthly)
    "CES4200000008": "Retail trade avg hourly earnings ($, SA)",
    # CPI for deflation
    "CPIAUCSL": "CPI-U all items (monthly, SA, 1982-84=100)",
    # Total retail sales
    "RSXFSN": "Advance retail sales ex food services (millions $, NSA)",
}

# BLS JOLTS series for retail trade (NAICS 44-45)
# Series IDs from BLS public data API
BLS_JOLTS_SERIES = {
    "JTS440000000000000TSR": "Total separations rate, retail trade",
    "JTS440000000000000QUR": "Quits rate, retail trade",
    "JTS440000000000000HIR": "Hires rate, retail trade",
}


def fetch_fred_series(series_id: str, start: str = "2008-01-01",
                      end: str = "2024-12-31",
                      api_key: str = None) -> list[dict]:
    """
    Fetch a FRED series. If no API key, falls back to scraping
    the CSV download endpoint.
    """
    # Use the CSV observation download (no API key needed)
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series_id}"
           f"&cosd={start}&coed={end}")
    req = Request(url)
    req.add_header("User-Agent", "WCT-Research academic-research@university.edu")

    try:
        with urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except (HTTPError, URLError) as e:
        print(f"  [WARN] Could not fetch FRED series {series_id}: {e}")
        return []

    # Parse CSV: DATE, VALUE
    observations = []
    for line in text.strip().split("\n")[1:]:  # skip header
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[1] != ".":
            try:
                observations.append({
                    "date": parts[0],
                    "value": float(parts[1]),
                })
            except ValueError:
                continue

    return observations


def aggregate_fred_annual(observations: list[dict]) -> dict:
    """Aggregate monthly/quarterly FRED observations to annual means."""
    from collections import defaultdict
    year_vals = defaultdict(list)
    for obs in observations:
        year = obs["date"][:4]
        year_vals[year].append(obs["value"])

    return {
        int(y): round(sum(v) / len(v), 2)
        for y, v in year_vals.items()
    }


def fetch_bls_jolts(output_dir: str = "data/raw") -> dict:
    """
    Fetch BLS JOLTS data for retail trade.

    Uses the BLS public data API v1 (no registration needed).
    Rate limit: 25 queries per 24 hours for unregistered users.
    """
    url = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
    series_ids = list(BLS_JOLTS_SERIES.keys())

    payload = json.dumps({
        "seriesid": series_ids,
        "startyear": "2008",
        "endyear": "2024",
    }).encode("utf-8")

    req = Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "WCT-Research academic-research@university.edu")

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError) as e:
        print(f"  [WARN] Could not fetch BLS JOLTS data: {e}")
        return {}

    # Parse response
    from collections import defaultdict
    annual = defaultdict(dict)

    if data.get("status") != "REQUEST_SUCCEEDED":
        print(f"  [WARN] BLS API returned status: {data.get('status')}")
        print(f"  [WARN] Message: {data.get('message', '')}")
        return {}

    for series in data.get("Results", {}).get("series", []):
        sid = series.get("seriesID", "")
        label = BLS_JOLTS_SERIES.get(sid, sid)

        # Map to short column name
        if "TSR" in sid:
            col = "turnover_rate"
        elif "QUR" in sid:
            col = "quits_rate"
        elif "HIR" in sid:
            col = "hires_rate"
        else:
            col = sid

        year_vals = defaultdict(list)
        for entry in series.get("data", []):
            year = int(entry["year"])
            try:
                val = float(entry["value"])
                year_vals[year].append(val)
            except (ValueError, KeyError):
                continue

        for year, vals in year_vals.items():
            annual[year][col] = round(sum(vals) / len(vals), 2)

    return dict(annual)


def build_macro_panel(output_dir: str = "data/raw") -> str:
    """Build the annual macro controls panel."""
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "macro_controls.csv")

    print("[MACRO] Fetching FRED series...")
    fred_data = {}
    for series_id, desc in FRED_SERIES.items():
        print(f"  Fetching {series_id}: {desc}")
        obs = fetch_fred_series(series_id)
        fred_data[series_id] = aggregate_fred_annual(obs)
        time.sleep(0.3)

    print("[MACRO] Fetching BLS JOLTS data...")
    jolts_data = fetch_bls_jolts()
    time.sleep(0.3)

    # Combine into annual panel
    all_years = set()
    for series_vals in fred_data.values():
        all_years.update(series_vals.keys())
    all_years.update(jolts_data.keys())
    years = sorted(y for y in all_years if 2008 <= y <= 2024)

    rows = []
    for year in years:
        row = {
            "year": year,
            "ecom_share_pct": fred_data.get("ECOMPCTSA", {}).get(year, ""),
            "retail_employment_k": fred_data.get("CES4200000001", {}).get(year, ""),
            "retail_avg_hourly_wage": fred_data.get("CES4200000008", {}).get(year, ""),
            "cpi": fred_data.get("CPIAUCSL", {}).get(year, ""),
            "retail_sales_m": fred_data.get("RSXFSN", {}).get(year, ""),
            "retail_turnover_rate": jolts_data.get(year, {}).get("turnover_rate", ""),
            "retail_quits_rate": jolts_data.get(year, {}).get("quits_rate", ""),
            "retail_hires_rate": jolts_data.get(year, {}).get("hires_rate", ""),
        }
        rows.append(row)

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "year", "ecom_share_pct", "retail_employment_k",
            "retail_avg_hourly_wage", "cpi", "retail_sales_m",
            "retail_turnover_rate", "retail_quits_rate", "retail_hires_rate",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[MACRO] Wrote {len(rows)} year observations "
          f"({years[0]}-{years[-1]}) to {outpath}")
    return outpath


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    build_macro_panel()
