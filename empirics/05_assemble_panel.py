#!/usr/bin/env python3
"""
05_assemble_panel.py — Merge all data sources into a single empirical panel.

Inputs (from data/raw/):
  - acsi_scores.csv         (firm × year: customer satisfaction)
  - glassdoor_panel.csv     (firm × year: employee satisfaction)
  - edgar_financials.csv    (firm × year: revenue, capex, SGA, employees)
  - macro_controls.csv      (year: e-commerce share, wages, turnover)

Output: data/processed/wct_empirical_panel.csv
  A clean, merged firm × year panel ready for regression analysis.

Key constructed variables:
  - acsi_score:         Customer satisfaction (T proxy), 0-100
  - glassdoor_overall:  Employee well-being (W proxy), 1-5
  - capex_intensity:    Capex / Revenue (K proxy)
  - sga_intensity:      SGA / Revenue (workforce investment proxy)
  - ecom_share_pct:     E-commerce as % of retail (F proxy, macro-level)
  - log_revenue:        Log of revenue (size control)
  - log_assets:         Log of total assets (size control)
  - log_employees:      Log of employee count (size control)
  - K_x_W:              Interaction: capex_intensity × glassdoor_overall

Usage:
  python 05_assemble_panel.py
"""

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(filepath: str) -> list[dict]:
    """Load a CSV file into a list of dicts."""
    if not os.path.exists(filepath):
        print(f"[WARN] File not found: {filepath}")
        return []
    with open(filepath, "r") as f:
        return list(csv.DictReader(f))


def safe_float(val: str) -> float | None:
    """Convert to float, returning None for missing/invalid values."""
    if val is None or val == "" or val == "NA" or val == ".":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_log(val: float | None) -> float | None:
    """Compute natural log, returning None for non-positive values."""
    if val is None or val <= 0:
        return None
    return round(math.log(val), 4)


def winsorise(values: list[float], lower: float = 0.01,
              upper: float = 0.99) -> list[float]:
    """Winsorise a list of values at the given percentiles."""
    valid = sorted(v for v in values if v is not None)
    if len(valid) < 10:
        return values
    lo = valid[int(len(valid) * lower)]
    hi = valid[int(len(valid) * upper)]
    return [
        max(lo, min(hi, v)) if v is not None else None
        for v in values
    ]


def assemble_panel(data_dir: str = "data",
                   output_dir: str = "data/processed") -> str:
    """Merge all sources into the final empirical panel."""
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    raw = os.path.join(data_dir, "raw")
    acsi = load_csv(os.path.join(raw, "acsi_scores.csv"))
    glassdoor = load_csv(os.path.join(raw, "glassdoor_panel.csv"))
    edgar = load_csv(os.path.join(raw, "edgar_financials.csv"))
    macro = load_csv(os.path.join(raw, "macro_controls.csv"))

    print(f"[PANEL] Loaded: ACSI={len(acsi)}, Glassdoor={len(glassdoor)}, "
          f"EDGAR={len(edgar)}, Macro={len(macro)} rows")

    # Index by (ticker, year) for merging
    acsi_idx = {}
    for row in acsi:
        key = (row.get("ticker", ""), row.get("year", ""))
        if key[0] and key[1]:
            acsi_idx[key] = row

    glassdoor_idx = {}
    for row in glassdoor:
        key = (row.get("ticker", ""), row.get("year", ""))
        if key[0] and key[1]:
            glassdoor_idx[key] = row

    edgar_idx = {}
    for row in edgar:
        key = (row.get("ticker", ""), row.get("year", ""))
        if key[0] and key[1]:
            edgar_idx[key] = row

    macro_idx = {}
    for row in macro:
        macro_idx[row.get("year", "")] = row

    # Determine universe: all (ticker, year) pairs present in EDGAR
    # (since financial data is the backbone)
    universe = sorted(edgar_idx.keys())
    print(f"[PANEL] Universe: {len(universe)} firm-year observations "
          f"from EDGAR")

    # --- Merge Validation ---
    # Check for duplicate keys in each source
    _dup_acsi = len(acsi) - len(acsi_idx)
    _dup_gd = len(glassdoor) - len(glassdoor_idx)
    _dup_ed = len(edgar) - len(edgar_idx)
    if _dup_acsi or _dup_gd or _dup_ed:
        print(f"[PANEL] WARNING: Duplicate (ticker,year) keys detected:")
        if _dup_acsi: print(f"  ACSI: {_dup_acsi} duplicates dropped (last wins)")
        if _dup_gd:   print(f"  Glassdoor: {_dup_gd} duplicates dropped")
        if _dup_ed:   print(f"  EDGAR: {_dup_ed} duplicates dropped")

    # Report pre-merge coverage
    acsi_tickers = set(k[0] for k in acsi_idx)
    gd_tickers = set(k[0] for k in glassdoor_idx)
    ed_tickers = set(k[0] for k in edgar_idx)
    all_tickers = acsi_tickers | gd_tickers | ed_tickers
    triple = acsi_tickers & gd_tickers & ed_tickers
    print(f"[PANEL] Merge coverage:")
    print(f"  ACSI firms:      {len(acsi_tickers)}")
    print(f"  Glassdoor firms: {len(gd_tickers)}")
    print(f"  EDGAR firms:     {len(ed_tickers)}")
    print(f"  All three:       {len(triple)} firms")
    print(f"  Union:           {len(all_tickers)} firms")

    # Merge
    merged = []
    n_complete = 0
    for ticker, year in universe:
        ed = edgar_idx.get((ticker, year), {})
        ac = acsi_idx.get((ticker, year), {})
        gl = glassdoor_idx.get((ticker, year), {})
        mc = macro_idx.get(year, {})

        revenue = safe_float(ed.get("revenue"))
        capex = safe_float(ed.get("capex"))
        sga = safe_float(ed.get("sga"))
        assets = safe_float(ed.get("assets"))
        employees = safe_float(ed.get("employees"))
        net_income = safe_float(ed.get("net_income"))
        acsi_score = safe_float(ac.get("acsi_score"))
        gd_overall = safe_float(gl.get("glassdoor_overall"))
        gd_worklife = safe_float(gl.get("glassdoor_worklife"))
        gd_review_n = safe_float(gl.get("review_count"))
        ecom = safe_float(mc.get("ecom_share_pct"))
        cpi = safe_float(mc.get("cpi"))
        turnover = safe_float(mc.get("retail_turnover_rate"))
        avg_wage = safe_float(mc.get("retail_avg_hourly_wage"))

        # Computed variables
        capex_int = round(capex / revenue, 4) if (capex and revenue) else None
        sga_int = round(sga / revenue, 4) if (sga and revenue) else None
        margin = round(net_income / revenue, 4) if (net_income is not None and revenue) else None

        # Interaction term: K × W
        k_x_w = round(capex_int * gd_overall, 4) if (capex_int and gd_overall) else None

        # Real revenue (CPI-deflated, base 2020)
        real_rev = round(revenue / (cpi / 258.81), 0) if (revenue and cpi) else None

        # Revenue per employee (productivity proxy)
        rev_per_emp = round(revenue / employees, 0) if (revenue and employees) else None

        has_core = (acsi_score is not None and gd_overall is not None
                    and capex_int is not None)
        if has_core:
            n_complete += 1

        row = {
            "company": ed.get("company", ""),
            "ticker": ticker,
            "year": year,
            # Customer satisfaction (T proxy)
            "acsi_score": acsi_score if acsi_score is not None else "",
            # Employee well-being (W proxy)
            "glassdoor_overall": gd_overall if gd_overall is not None else "",
            "glassdoor_worklife": gd_worklife if gd_worklife is not None else "",
            "glassdoor_review_count": int(gd_review_n) if gd_review_n else "",
            # Financial variables
            "revenue": revenue if revenue is not None else "",
            "log_revenue": safe_log(revenue) if revenue else "",
            "capex": capex if capex is not None else "",
            "capex_intensity": capex_int if capex_int is not None else "",
            "sga": sga if sga is not None else "",
            "sga_intensity": sga_int if sga_int is not None else "",
            "assets": assets if assets is not None else "",
            "log_assets": safe_log(assets) if assets else "",
            "employees": int(employees) if employees else "",
            "log_employees": safe_log(employees) if employees else "",
            "net_income": net_income if net_income is not None else "",
            "net_margin": margin if margin is not None else "",
            "rev_per_employee": rev_per_emp if rev_per_emp else "",
            "real_revenue": real_rev if real_rev else "",
            # Interaction terms
            "K_x_W": k_x_w if k_x_w is not None else "",
            # Macro controls
            "ecom_share_pct": ecom if ecom is not None else "",
            "cpi": cpi if cpi is not None else "",
            "retail_turnover_rate": turnover if turnover is not None else "",
            "retail_avg_wage": avg_wage if avg_wage is not None else "",
        }
        merged.append(row)

    # Write output
    outpath = os.path.join(output_dir, "wct_empirical_panel.csv")
    fieldnames = list(merged[0].keys()) if merged else []
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    tickers = set(r["ticker"] for r in merged)
    years = sorted(set(r["year"] for r in merged))
    print(f"\n[PANEL] Final panel: {len(merged)} observations, "
          f"{len(tickers)} firms, {years[0]}-{years[-1]}")
    print(f"[PANEL] Complete observations (ACSI + Glassdoor + Capex): "
          f"{n_complete} ({100*n_complete/max(len(merged),1):.0f}%)")
    print(f"[PANEL] Output: {outpath}")

    # Summary statistics
    _print_summary(merged)

    return outpath


def _print_summary(panel: list[dict]):
    """Print summary statistics for key variables."""
    vars_to_summarise = [
        ("acsi_score", "ACSI Score (T proxy)"),
        ("glassdoor_overall", "Glassdoor Overall (W proxy)"),
        ("capex_intensity", "Capex/Revenue (K proxy)"),
        ("sga_intensity", "SGA/Revenue"),
        ("net_margin", "Net Margin"),
        ("K_x_W", "K × W Interaction"),
        ("ecom_share_pct", "E-commerce Share %"),
    ]

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"{'Variable':<30} {'N':>5} {'Mean':>8} {'SD':>8} "
          f"{'Min':>8} {'Max':>8}")
    print("-"*60)

    for var, label in vars_to_summarise:
        vals = [safe_float(r.get(var, "")) for r in panel]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        sd = (sum((v - mean)**2 for v in vals) / max(n - 1, 1)) ** 0.5
        lo = min(vals)
        hi = max(vals)
        print(f"{label:<30} {n:>5} {mean:>8.3f} {sd:>8.3f} "
              f"{lo:>8.3f} {hi:>8.3f}")

    print("="*60)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    assemble_panel()
