#!/usr/bin/env python3
"""
14_shift_project.py — Shift Project schedule stability test for WCT model.

Tests whether workers at high-digitalization retailers report worse schedule
stability (shorter advance notice, higher hours volatility) than workers at
lower-digitalization retailers.

DATA ACCESS:
  The Shift Project survey data is hosted on Harvard Dataverse:
    DOI: https://doi.org/10.7910/DVN/IL6SS2
    Title: "Shift Project National Survey Cross-sections"
    Deposited: 2025-06-27
    Files:
      - Shift_Codebook.html (1.7 MB) — public
      - Shift_Data.tab (49.4 MB) — RESTRICTED (requires access request)

  To request access:
    1. Visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IL6SS2
    2. Click "Request Access" for Shift_Data.tab
    3. Once approved, download and place at:
       data/shift_project/Shift_Data.tab

  The public codebook (already in repo at data/shift_project/Shift_Codebook.html)
  documents all variables. Key public variables for this test:
    - q1_employer: employer name (Walmart, Target, Kroger, etc.)
    - advance_notice_days: schedule advance notice
    - usualhours_clean: usual weekly hours
    - greatesthr_clean3: greatest weekly hours
    - leasthr_clean3: least weekly hours
    - wage_hourly: hourly wage indicator

  Technology/automation variables (leaderboard_stress, tech_selfcheckout,
  employer_trusts_you, etc.) are ALL restricted — not in the public release.

WHAT THIS SCRIPT TESTS (if data available):
  Workers at high-digitalization retailers (Walmart, Target, Kroger) vs.
  lower-digitalization retailers (Costco, Publix) — do they report shorter
  advance notice and higher hours volatility?

  This is a cross-sectional comparison, not a causal test. It provides
  mechanism evidence consistent with the Front-Stage Trap if high-digi
  workers report worse schedule conditions.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)

DATA_PATH = project_root / 'data' / 'shift_project' / 'Shift_Data.tab'
DATA_PATH_CSV = project_root / 'data' / 'shift_project' / 'Shift_Data.csv'

# Digitalization tiers based on our digitalization_score_by_year.csv
# (2020 midpoint: Walmart 75%, Target 71%, Kroger 50% vs Costco 40%)
HIGH_DIGI_EMPLOYERS = ['walmart', 'target', 'kroger']
LOW_DIGI_EMPLOYERS = ['costco', 'publix']


def normalize_employer(name):
    """Normalize employer name for matching."""
    if pd.isna(name):
        return None
    name = str(name).lower().strip()
    for brand in HIGH_DIGI_EMPLOYERS + LOW_DIGI_EMPLOYERS:
        if brand in name:
            return brand
    return None


def main():
    print("=" * 70)
    print("SHIFT PROJECT: SCHEDULE STABILITY TEST")
    print("=" * 70)
    print(f"DOI: https://doi.org/10.7910/DVN/IL6SS2")
    print(f"Source: Harvard Dataverse — Shift Project National Survey Cross-sections\n")

    # Check for data file
    data_path = None
    for p in [DATA_PATH, DATA_PATH_CSV]:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        print("DATA NOT FOUND.")
        print(f"  Expected at: {DATA_PATH}")
        print(f"           or: {DATA_PATH_CSV}")
        print()
        print("  The Shift Project survey data is RESTRICTED on Harvard Dataverse.")
        print("  To access:")
        print("    1. Visit https://dataverse.harvard.edu/dataset.xhtml?"
              "persistentId=doi:10.7910/DVN/IL6SS2")
        print("    2. Click 'Request Access' for Shift_Data.tab")
        print("    3. Once approved, download and place at:")
        print(f"       {DATA_PATH}")
        print()
        print("  The codebook is already in the repo at:")
        print("    data/shift_project/Shift_Codebook.html")
        print("    data/shift_project/codebook_variables.csv")
        print()
        print("  NOTE: All 167 technology/automation variables (leaderboard_stress,")
        print("  tech exposure, monitoring, employer_trusts_you) are restricted")
        print("  even within the dataset. Only 78 demographic/schedule/financial")
        print("  variables are in the public release.")
        print()
        print("[SKIP] Cannot run tests without data. Exiting.")
        return

    # Load data
    print(f"Loading data from: {data_path}")
    sep = '\t' if str(data_path).endswith('.tab') else ','
    try:
        df = pd.read_csv(data_path, sep=sep, low_memory=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # Identify employer column
    emp_col = None
    for candidate in ['q1_employer', 'employer', 'company', 'employer_name']:
        if candidate in df.columns:
            emp_col = candidate
            break

    if emp_col is None:
        print(f"  Cannot find employer column. Available columns: {list(df.columns[:20])}")
        return

    print(f"  Employer column: {emp_col}")
    df['brand'] = df[emp_col].apply(normalize_employer)
    df = df.dropna(subset=['brand'])
    print(f"  Matched to study brands: {len(df)} workers")
    print(f"  Brand distribution:")
    for brand, count in df['brand'].value_counts().items():
        tier = 'HIGH-DIGI' if brand in HIGH_DIGI_EMPLOYERS else 'LOW-DIGI'
        print(f"    {brand}: {count} ({tier})")

    df['high_digi'] = df['brand'].isin(HIGH_DIGI_EMPLOYERS).astype(int)

    # Test available schedule variables
    schedule_vars = ['advance_notice_days', 'advance_notice_numeric',
                     'usualhours_clean', 'greatesthr_clean3', 'leasthr_clean3']

    available_vars = [v for v in schedule_vars if v in df.columns]
    if not available_vars:
        print(f"\n  No schedule variables found in data.")
        print(f"  Available columns (first 30): {list(df.columns[:30])}")
        return

    # Hours volatility = greatest - least weekly hours
    if 'greatesthr_clean3' in df.columns and 'leasthr_clean3' in df.columns:
        df['hours_volatility'] = df['greatesthr_clean3'] - df['leasthr_clean3']
        available_vars.append('hours_volatility')

    print(f"\n  Available schedule variables: {available_vars}")

    # Simple comparison tests
    from scipy.stats import ttest_ind, mannwhitneyu

    print(f"\n{'='*60}")
    print("RESULTS: High-Digi vs. Low-Digi Employer Comparison")
    print(f"{'='*60}")

    for var in available_vars:
        sub = df.dropna(subset=[var])
        high = sub[sub['high_digi'] == 1][var]
        low = sub[sub['high_digi'] == 0][var]

        if len(high) < 10 or len(low) < 10:
            print(f"\n  {var}: insufficient data (high={len(high)}, low={len(low)})")
            continue

        t_stat, p_val = ttest_ind(high, low, equal_var=False)
        u_stat, p_mw = mannwhitneyu(high, low, alternative='two-sided')

        print(f"\n  {var}:")
        print(f"    High-digi (N={len(high)}): mean={high.mean():.2f}, SD={high.std():.2f}")
        print(f"    Low-digi  (N={len(low)}):  mean={low.mean():.2f}, SD={low.std():.2f}")
        print(f"    Diff: {high.mean() - low.mean():.2f}")
        print(f"    Welch t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"    Mann-Whitney: p={p_mw:.4f}")

        if var == 'advance_notice_days' and high.mean() < low.mean():
            print(f"    → High-digi workers get LESS advance notice (consistent with W depletion)")
        elif var == 'hours_volatility' and high.mean() > low.mean():
            print(f"    → High-digi workers have MORE hours volatility (consistent with W depletion)")

    print(f"\n  CAVEAT: Cross-sectional comparison, not causal. Selection bias")
    print(f"  possible (workers self-select into employers). Technology/automation")
    print(f"  variables (the most relevant for mechanism testing) are restricted.")

    print("\n[DONE] Shift Project analysis complete.")


if __name__ == '__main__':
    main()
