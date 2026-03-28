#!/usr/bin/env python3
"""
07_alternative_specs.py — Creative alternative specifications for WCT tests.

Diagnostic findings from the baseline:
  1. Glassdoor within-firm SD is 30% of total — firm FE absorbs the signal
  2. Cross-sectional Corr(ACSI, Glassdoor) = 0.747 — signal is between firms
  3. Capex is a noisy K proxy; need to distinguish digital from physical

Alternative approaches:
  A. Between-effects & Mundlak CRE — capture cross-sectional complementarity
  B. Capex/SGA ratio as f (allocation tilt) — the model's own variable
  C. Lagged specifications — tech investment takes time to affect outcomes
  D. COVID controls — 2020-2021 exogenous shock
  E. Revenue growth as control — firm momentum
  F. Employee-related liabilities from EDGAR — financial W proxy
  G. Split sample by W level — test whether K helps only when W is high
  H. Regime-switching — the model predicts threshold effects, not linear
  I. Alternative outcome: net margin — test through profitability channel
  J. Pooled OLS with industry/time controls — if signal is cross-sectional

Usage:
  python 07_alternative_specs.py
"""

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from linearmodels.panel import (
        PanelOLS, RandomEffects, BetweenOLS, PooledOLS,
    )
    HAS_LM = True
except ImportError:
    HAS_LM = False


# -----------------------------------------------------------------------
# 0. Enhanced data: fetch EmployeeRelatedLiabilities from EDGAR
# -----------------------------------------------------------------------

SEC_USER_AGENT = "WCT-Research academic-research@university.edu"

RETAIL_CIKS = {
    "WMT": 104169, "TGT": 27419, "COST": 909832, "JWN": 72333,
    "M": 794367, "HD": 354950, "LOW": 60667, "BBY": 764478,
    "AMZN": 1018724, "KR": 56873, "CVS": 64803, "WBA": 1618921,
    "DG": 34067, "DLTR": 935703, "KSS": 885639, "TJX": 109198,
    "GPS": 39911,
}

EMP_LIABILITY_TAGS = [
    "EmployeeRelatedLiabilitiesCurrent",
    "EmployeeRelatedLiabilitiesCurrentAndNoncurrent",
    "AccruedSalariesCurrentAndNoncurrent",
    "AccruedSalariesCurrent",
    "EmployeeBenefitsAndShareBasedCompensation",
]


def fetch_employee_liabilities() -> dict:
    """Fetch employee-related liabilities from SEC EDGAR."""
    results = {}  # (ticker, year) -> value
    for ticker, cik in RETAIL_CIKS.items():
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        req = Request(url)
        req.add_header("User-Agent", SEC_USER_AGENT)
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError):
            continue

        gaap = data.get("facts", {}).get("us-gaap", {})
        for tag in EMP_LIABILITY_TAGS:
            tag_data = gaap.get(tag, {})
            for unit_key, entries in tag_data.get("units", {}).items():
                if unit_key != "USD":
                    continue
                for e in entries:
                    if e.get("form") not in ("10-K", "10-K/A"):
                        continue
                    end = e.get("end", "")
                    if len(end) >= 4:
                        try:
                            fy = int(end[:4])
                        except ValueError:
                            continue
                        if 2008 <= fy <= 2024:
                            key = (ticker, fy)
                            if key not in results:
                                results[key] = e.get("val", 0)
        time.sleep(0.15)
    return results


# -----------------------------------------------------------------------
# 1. Load and enrich panel
# -----------------------------------------------------------------------

def load_and_enrich(panel_path: str = "data/processed/wct_empirical_panel.csv",
                    fetch_liabilities: bool = True) -> pd.DataFrame:
    """Load panel and add creative variables."""
    df = pd.read_csv(panel_path)
    num_cols = [c for c in df.columns if c not in ("company", "ticker")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- A. Capex/SGA ratio (f proxy: allocation tilt toward tech) ---
    df["capex_sga_ratio"] = df["capex"] / df["sga"]
    df["log_capex_sga"] = np.log(df["capex_sga_ratio"].clip(lower=0.001))

    # --- B. Lagged variables ---
    df = df.sort_values(["ticker", "year"])
    for var in ["capex_intensity", "glassdoor_overall", "sga_intensity",
                "acsi_score", "capex_sga_ratio"]:
        df[f"{var}_lag1"] = df.groupby("ticker")[var].shift(1)

    # --- C. Revenue growth (momentum control) ---
    df["rev_growth"] = df.groupby("ticker")["revenue"].pct_change()

    # --- D. COVID indicator ---
    df["covid"] = ((df["year"] == 2020) | (df["year"] == 2021)).astype(int)

    # --- E. Firm-mean variables for Mundlak correction ---
    for var in ["capex_intensity", "glassdoor_overall", "sga_intensity",
                "capex_sga_ratio"]:
        df[f"{var}_bar"] = df.groupby("ticker")[var].transform("mean")

    # --- F. Employee-related liabilities from EDGAR ---
    if fetch_liabilities:
        print("[EDGAR] Fetching employee-related liabilities...")
        emp_liab = fetch_employee_liabilities()
        df["emp_liabilities"] = df.apply(
            lambda r: emp_liab.get((r["ticker"], r["year"])), axis=1
        )
        df["emp_liab_rev"] = df["emp_liabilities"] / df["revenue"]
        print(f"  Got {df['emp_liabilities'].notna().sum()} obs with "
              f"employee liabilities data")
    else:
        df["emp_liabilities"] = np.nan
        df["emp_liab_rev"] = np.nan

    # --- G. High-W / Low-W partition ---
    gd_median = df["glassdoor_overall"].median()
    df["high_W"] = (df["glassdoor_overall"] >= gd_median).astype(int)
    print(f"  Glassdoor median: {gd_median:.2f} "
          f"(high_W={df['high_W'].sum()}, low_W={(1-df['high_W']).sum()})")

    # --- H. Standardized variables ---
    for var in ["capex_intensity", "glassdoor_overall", "sga_intensity",
                "capex_sga_ratio", "emp_liab_rev"]:
        mu = df[var].mean()
        sd = df[var].std()
        if sd > 0:
            df[f"{var}_z"] = (df[var] - mu) / sd

    # --- I. Interaction terms ---
    df["K_x_W_z"] = df["capex_intensity_z"] * df["glassdoor_overall_z"]
    df["f_x_W"] = df["capex_sga_ratio"] * df["glassdoor_overall"]
    df["f_x_W_z"] = df.get("capex_sga_ratio_z", 0) * df.get("glassdoor_overall_z", 0)
    df["K_lag_x_W_lag"] = df["capex_intensity_lag1"] * df["glassdoor_overall_lag1"]

    # Employee liability × capex interaction
    df["empliab_x_K"] = df["emp_liab_rev"] * df["capex_intensity"]

    # Triple: K × W × covid
    df["K_x_W_x_covid"] = df["K_x_W_z"] * df["covid"]

    return df


def _sig(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""


def run_spec(df, y_col, x_cols, method="fe", label="", cluster="ticker"):
    """Run a specification and return formatted results."""
    cols = [y_col] + x_cols + [cluster, "year"]
    sub = df[cols].dropna()
    n = len(sub)
    n_firms = sub[cluster].nunique()
    if n < 30:
        print(f"  [{label}] N={n} — skipping (too few obs)")
        return None

    panel = sub.set_index([cluster, "year"])
    y = panel[y_col]
    X = panel[x_cols]

    if method == "fe" and HAS_LM:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True,
                       check_rank=False)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        r2_label = f"R2w={res.rsquared_within:.3f}"
    elif method == "be" and HAS_LM:
        mod = BetweenOLS(y, sm.add_constant(X))
        res = mod.fit()
        r2_label = f"R2b={res.rsquared:.3f}"
    elif method == "re" and HAS_LM:
        mod = RandomEffects(y, sm.add_constant(X))
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        r2_label = f"R2={res.rsquared:.3f}"
    elif method == "mundlak" and HAS_LM:
        # Mundlak CRE: RE with firm-mean regressors
        mod = RandomEffects(y, sm.add_constant(X))
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        r2_label = f"R2={res.rsquared:.3f}"
    elif method == "pooled":
        X_c = sm.add_constant(X)
        mod = sm.OLS(y, X_c)
        groups = sub[cluster].astype("category").cat.codes.values
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
        r2_label = f"R2={res.rsquared:.3f}"
    else:
        # Fallback to pooled OLS
        X_c = sm.add_constant(X)
        mod = sm.OLS(y, X_c)
        groups = sub[cluster].astype("category").cat.codes.values
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
        r2_label = f"R2={res.rsquared:.3f}"

    print(f"\n  [{label}] N={n}, firms={n_firms}, {r2_label}")
    for var in x_cols:
        if var in res.params.index:
            b = res.params[var]
            se = res.std_errors[var] if hasattr(res, 'std_errors') else res.bse[var]
            t = res.tstats[var] if hasattr(res, 'tstats') else res.tvalues[var]
            p = res.pvalues[var]
            print(f"    {var:35s}  b={b:>9.4f}  se={se:.4f}  "
                  f"t={t:>6.3f}  p={p:.4f} {_sig(p)}")

    return res


# -----------------------------------------------------------------------
# Alternative Specifications
# -----------------------------------------------------------------------

def spec_A_between_effects(df):
    """A. Between-effects estimator — cross-sectional complementarity."""
    print("\n" + "="*70)
    print("SPEC A: BETWEEN-EFFECTS (cross-sectional complementarity)")
    print("  Rationale: the K×W signal is cross-sectional (Costco vs DG).")
    print("  Firm FE absorbs this. Between-effects recovers it.")
    print("="*70)

    run_spec(df, "acsi_score",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "log_revenue"],
             method="be", label="A1: Between OLS")


def spec_B_mundlak_cre(df):
    """B. Mundlak CRE — decomposes between and within effects."""
    print("\n" + "="*70)
    print("SPEC B: MUNDLAK CORRELATED RANDOM EFFECTS")
    print("  Rationale: captures both between-firm and within-firm variation.")
    print("  Firm means (x_bar) capture the cross-sectional relationship.")
    print("="*70)

    run_spec(df, "acsi_score",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "capex_intensity_bar", "glassdoor_overall_bar",
              "log_revenue"],
             method="mundlak", label="B1: Mundlak CRE")


def spec_C_allocation_ratio(df):
    """C. Capex/SGA ratio as f (the model's allocation variable)."""
    print("\n" + "="*70)
    print("SPEC C: CAPEX/SGA RATIO AS f (ALLOCATION TILT)")
    print("  Rationale: The model's f = share allocated to tech vs workforce.")
    print("  Capex/SGA directly measures this allocation decision.")
    print("="*70)

    # C1: f × W interaction with firm FE
    run_spec(df, "acsi_score",
             ["capex_sga_ratio_z", "glassdoor_overall_z", "f_x_W_z",
              "log_revenue"],
             method="fe", label="C1: f×W with firm+year FE")

    # C2: Between-effects with f ratio
    run_spec(df, "acsi_score",
             ["capex_sga_ratio_z", "glassdoor_overall_z", "f_x_W_z",
              "log_revenue"],
             method="be", label="C2: f×W between-effects")

    # C3: Quadratic in f (parallel to Front-Stage Trap)
    df["f_ratio_sq"] = df["capex_sga_ratio"] ** 2
    run_spec(df, "acsi_score",
             ["capex_sga_ratio", "f_ratio_sq", "glassdoor_overall",
              "log_revenue"],
             method="fe", label="C3: Quadratic f (firm FE)")


def spec_D_lagged(df):
    """D. Lagged specifications — investment takes time."""
    print("\n" + "="*70)
    print("SPEC D: LAGGED SPECIFICATIONS")
    print("  Rationale: Capex in year t affects customer experience in t+1.")
    print("  Glassdoor changes propagate with delay to customer satisfaction.")
    print("="*70)

    # Standardize lagged variables
    for var in ["capex_intensity_lag1", "glassdoor_overall_lag1"]:
        mu = df[var].mean()
        sd = df[var].std()
        if sd and sd > 0:
            df[f"{var}_z"] = (df[var] - mu) / sd
    df["K_lag_x_W_lag_z"] = (df.get("capex_intensity_lag1_z", 0) *
                              df.get("glassdoor_overall_lag1_z", 0))

    # D1: Lagged K and W with firm FE
    run_spec(df, "acsi_score",
             ["capex_intensity_lag1_z", "glassdoor_overall_lag1_z",
              "K_lag_x_W_lag_z", "log_revenue"],
             method="fe", label="D1: Lagged K×W (firm+year FE)")

    # D2: Lagged between-effects
    run_spec(df, "acsi_score",
             ["capex_intensity_lag1_z", "glassdoor_overall_lag1_z",
              "K_lag_x_W_lag_z", "log_revenue"],
             method="be", label="D2: Lagged K×W (between)")


def spec_E_covid_controls(df):
    """E. With COVID controls and revenue growth."""
    print("\n" + "="*70)
    print("SPEC E: COVID CONTROLS + REVENUE GROWTH")
    print("  Rationale: 2020-21 was a massive exogenous shock to both")
    print("  workforce well-being and digitalisation (forced e-commerce).")
    print("="*70)

    # E1: Baseline + COVID dummy
    run_spec(df, "acsi_score",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "covid", "log_revenue"],
             method="fe", label="E1: + COVID dummy (firm FE)")

    # E2: Exclude COVID years entirely
    df_nocovid = df[~df["year"].isin([2020, 2021])].copy()
    run_spec(df_nocovid, "acsi_score",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "log_revenue"],
             method="fe", label="E2: Exclude 2020-21 (firm FE)")

    # E3: Between-effects excluding COVID
    run_spec(df_nocovid, "acsi_score",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "log_revenue"],
             method="be", label="E3: Exclude 2020-21 (between)")

    # E4: + Revenue growth control
    run_spec(df, "acsi_score",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "rev_growth", "covid", "log_revenue"],
             method="fe", label="E4: + Rev growth + COVID (firm FE)")


def spec_F_emp_liabilities(df):
    """F. Employee liabilities as financial W proxy."""
    print("\n" + "="*70)
    print("SPEC F: EMPLOYEE-RELATED LIABILITIES (financial W proxy)")
    print("  Rationale: EmpLiab/Revenue captures actual $ spent on workforce")
    print("  (wages, benefits, accrued compensation). Has within-firm variation.")
    print("="*70)

    n_valid = df["emp_liab_rev"].notna().sum()
    if n_valid < 30:
        print(f"  Only {n_valid} obs with employee liabilities — skipping.")
        return

    # Standardize
    mu = df["emp_liab_rev"].mean()
    sd = df["emp_liab_rev"].std()
    if sd > 0:
        df["emp_liab_z"] = (df["emp_liab_rev"] - mu) / sd
    df["K_x_empliab_z"] = df["capex_intensity_z"] * df.get("emp_liab_z", 0)

    # F1: Emp liability as W proxy (firm FE)
    run_spec(df, "acsi_score",
             ["capex_intensity_z", "emp_liab_z", "K_x_empliab_z",
              "log_revenue"],
             method="fe", label="F1: K×EmpLiab (firm+year FE)")

    # F2: Between-effects
    run_spec(df, "acsi_score",
             ["capex_intensity_z", "emp_liab_z", "K_x_empliab_z",
              "log_revenue"],
             method="be", label="F2: K×EmpLiab (between)")


def spec_G_split_sample(df):
    """G. Split sample by W level — test if K helps only when W is high."""
    print("\n" + "="*70)
    print("SPEC G: SPLIT SAMPLE BY WORKFORCE WELL-BEING (W)")
    print("  Rationale: The model predicts complementarity ABOVE a threshold.")
    print("  Below the burnout cliff, K investment may hurt regardless.")
    print("="*70)

    df_high = df[df["high_W"] == 1].copy()
    df_low = df[df["high_W"] == 0].copy()

    print(f"\n  High-W firms (Glassdoor >= median): {df_high.ticker.nunique()} firms")
    print(f"  Low-W firms  (Glassdoor <  median): {df_low.ticker.nunique()} firms")

    # G1: K → ACSI for high-W firms only
    run_spec(df_high, "acsi_score",
             ["capex_intensity_z", "log_revenue"],
             method="fe", label="G1: K→ACSI (high-W only, FE)")

    # G2: K → ACSI for low-W firms only
    run_spec(df_low, "acsi_score",
             ["capex_intensity_z", "log_revenue"],
             method="fe", label="G2: K→ACSI (low-W only, FE)")

    # G3: Between-effects, high-W
    run_spec(df_high, "acsi_score",
             ["capex_intensity_z", "log_revenue"],
             method="be", label="G3: K→ACSI (high-W, between)")

    # G4: Between-effects, low-W
    run_spec(df_low, "acsi_score",
             ["capex_intensity_z", "log_revenue"],
             method="be", label="G4: K→ACSI (low-W, between)")


def spec_H_profitability(df):
    """H. Alternative outcome: net margin — profitability channel."""
    print("\n" + "="*70)
    print("SPEC H: PROFITABILITY CHANNEL (Net Margin as outcome)")
    print("  Rationale: If K×W complementarity exists, it should also")
    print("  show up in profitability, not just customer satisfaction.")
    print("="*70)

    run_spec(df, "net_margin",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "log_revenue"],
             method="fe", label="H1: K×W→Margin (firm+year FE)")

    run_spec(df, "net_margin",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "log_revenue"],
             method="be", label="H2: K×W→Margin (between)")


def spec_I_pooled_with_industry(df):
    """I. Pooled OLS with industry controls — cross-sectional."""
    print("\n" + "="*70)
    print("SPEC I: POOLED OLS WITH INDUSTRY CONTROLS")
    print("  Rationale: If the signal is cross-sectional, pooled OLS")
    print("  with cluster-robust SE may be more appropriate than FE.")
    print("="*70)

    run_spec(df, "acsi_score",
             ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z",
              "log_revenue", "ecom_share_pct"],
             method="pooled", label="I1: Pooled OLS (cluster-robust)")


def spec_J_ecom_enhanced(df):
    """J. Enhanced Front-Stage Trap with workforce moderation."""
    print("\n" + "="*70)
    print("SPEC J: ENHANCED FRONT-STAGE TRAP")
    print("  Rationale: The model predicts that the Front-Stage Trap")
    print("  is MODERATED by workforce investment — good jobs raise F_crit.")
    print("  Test: ecom² × Glassdoor interaction (triple).")
    print("="*70)

    df["ecom2"] = df["ecom_share_pct"] ** 2
    df["ecom2_x_gd"] = df["ecom2"] * df["glassdoor_overall"]
    df["ecom_x_gd"] = df["ecom_share_pct"] * df["glassdoor_overall"]

    # J1: Base quadratic (replicate Test 2)
    run_spec(df, "acsi_score",
             ["ecom_share_pct", "ecom2", "log_revenue"],
             method="fe", label="J1: Quadratic ecom (firm FE)")

    # J2: + Glassdoor moderation of the trap
    run_spec(df, "acsi_score",
             ["ecom_share_pct", "ecom2", "glassdoor_overall",
              "ecom2_x_gd", "log_revenue"],
             method="fe", label="J2: ecom² × Glassdoor (firm FE)")

    # J3: Between-effects version
    run_spec(df, "acsi_score",
             ["ecom_share_pct", "ecom2", "glassdoor_overall",
              "ecom2_x_gd", "log_revenue"],
             method="be", label="J3: ecom² × Glassdoor (between)")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    df = load_and_enrich(fetch_liabilities=True)

    print(f"\n[DATA] Enriched panel: {len(df)} obs, {df.ticker.nunique()} firms")
    print(f"[DATA] Complete K×W obs: "
          f"{df[['acsi_score','glassdoor_overall','capex_intensity']].dropna().shape[0]}")
    print(f"[DATA] Capex/SGA ratio obs: {df['capex_sga_ratio'].notna().sum()}")
    print(f"[DATA] Emp liabilities obs: {df['emp_liabilities'].notna().sum()}")

    spec_A_between_effects(df)
    spec_B_mundlak_cre(df)
    spec_C_allocation_ratio(df)
    spec_D_lagged(df)
    spec_E_covid_controls(df)
    spec_F_emp_liabilities(df)
    spec_G_split_sample(df)
    spec_H_profitability(df)
    spec_I_pooled_with_industry(df)
    spec_J_ecom_enhanced(df)

    # ---- Summary of key findings ----
    print("\n" + "="*70)
    print("SUMMARY OF ALTERNATIVE SPECIFICATIONS")
    print("="*70)
    print("""
Key questions answered:
  1. Is complementarity (p>0) detectable BETWEEN firms?     → Check Spec A, I
  2. Does the allocation ratio (f=capex/SGA) work better?   → Check Spec C
  3. Do lagged effects improve the signal?                  → Check Spec D
  4. Is COVID driving the negative within-firm result?      → Check Spec E
  5. Does emp liability (financial W) work as W proxy?      → Check Spec F
  6. Does K help ACSI only for high-W firms?                → Check Spec G
  7. Does K×W predict profitability too?                    → Check Spec H
  8. Does good workforce raise the Front-Stage threshold?   → Check Spec J
    """)


if __name__ == "__main__":
    main()
