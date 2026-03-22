#!/usr/bin/env python3
"""
eu_02_assemble_and_test.py — Assemble European panel and run replication tests.

This script:
  1. Merges Trustpilot + Glassdoor + yfinance financials + macro controls
  2. Validates merge quality (coverage, missingness, overlap diagnostics)
  3. Runs the three core WCT tests (complementarity, Front-Stage Trap, Good Jobs)
  4. Runs full robustness battery
  5. Compares European results with US results

Key methodological differences from US panel:
  - Customer satisfaction: Trustpilot (complaint-biased, 1-5 scale) vs ACSI (survey, 0-100)
  - Geography: 6 countries vs US-only
  - Currency: Mixed (GBP, EUR, SEK) — use intensity ratios, not levels
  - E-commerce context: Higher baseline penetration in UK/NL than US

Usage:
  python eu_02_assemble_and_test.py
"""

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

try:
    from linearmodels.panel import PanelOLS, BetweenOLS, RandomEffects
    HAS_LM = True
except ImportError:
    HAS_LM = False


# -----------------------------------------------------------------------
# 1. Panel Assembly with Robust Merging
# -----------------------------------------------------------------------

def assemble_panel():
    """Merge all European data sources with validation."""
    print("="*70)
    print("PHASE 1: PANEL ASSEMBLY WITH MERGE DIAGNOSTICS")
    print("="*70)

    # Load raw data
    trust = pd.read_csv("data/raw/eu_trustpilot.csv")
    gd = pd.read_csv("data/raw/eu_glassdoor.csv")
    fin = pd.read_csv("data/raw/eu_financials.csv")
    macro = pd.read_csv("data/raw/eu_macro.csv")

    print(f"\n  Raw data loaded:")
    print(f"    Trustpilot:  {len(trust)} obs, {trust.company.nunique()} firms")
    print(f"    Glassdoor:   {len(gd)} obs, {gd.company.nunique()} firms")
    print(f"    Financials:  {len(fin)} obs, {fin.ticker.nunique()} firms")
    print(f"    Macro:       {len(macro)} years")

    # Convert types
    for df in [trust, gd, fin]:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in fin.columns:
        if col not in ("company", "ticker", "country"):
            fin[col] = pd.to_numeric(fin[col], errors="coerce")

    # --- Merge Step 1: Financials as backbone ---
    panel = fin.copy()

    # --- Merge Step 2: Trustpilot (T proxy) ---
    trust_cols = trust[["ticker", "year", "trustpilot_score",
                        "trustpilot_reviews"]].copy()
    panel = panel.merge(trust_cols, on=["ticker", "year"], how="left")

    # --- Merge Step 3: Glassdoor (W proxy) ---
    gd_cols = gd[["ticker", "year", "glassdoor_overall",
                  "review_count"]].copy()
    gd_cols = gd_cols.rename(columns={"review_count": "gd_review_count"})
    panel = panel.merge(gd_cols, on=["ticker", "year"], how="left")

    # --- Merge Step 4: Macro controls ---
    macro["year"] = pd.to_numeric(macro["year"], errors="coerce")
    for col in macro.columns:
        if col != "year":
            macro[col] = pd.to_numeric(macro[col], errors="coerce")

    # Use UK internet sales for UK firms, EU ecom share for others
    panel = panel.merge(macro, on="year", how="left")
    panel["ecom_share"] = np.where(
        panel["country"] == "GB",
        panel["uk_internet_sales_pct"],
        panel["eu_ecom_share_pct"]
    )

    # --- Computed variables ---
    panel["log_revenue"] = np.log(panel["revenue"].clip(lower=1))
    panel["log_assets"] = np.log(panel["assets"].clip(lower=1))
    panel["net_margin"] = panel["net_income"] / panel["revenue"]
    panel["ecom2"] = panel["ecom_share"] ** 2

    # Standardize
    for v in ["capex_intensity", "glassdoor_overall", "trustpilot_score",
              "sga_intensity"]:
        mu = panel[v].mean()
        sd = panel[v].std()
        if sd and sd > 0:
            panel[f"{v}_z"] = (panel[v] - mu) / sd

    # Interactions
    panel["K_x_W_z"] = (panel.get("capex_intensity_z", 0) *
                         panel.get("glassdoor_overall_z", 0))
    panel["K_x_T_z"] = (panel.get("capex_intensity_z", 0) *
                         panel.get("trustpilot_score_z", 0))

    # Lags
    panel = panel.sort_values(["ticker", "year"])
    for v in ["capex_intensity_z", "glassdoor_overall_z", "trustpilot_score"]:
        panel[f"{v}_L1"] = panel.groupby("ticker")[v].shift(1)
    panel["K_x_W_z_L1"] = (panel["capex_intensity_z_L1"] *
                            panel["glassdoor_overall_z_L1"])

    # COVID
    panel["covid"] = panel["year"].isin([2020, 2021]).astype(int)

    # Time trend
    panel["time_trend"] = panel["year"] - 2019

    # Firm means (Mundlak)
    for v in ["capex_intensity", "glassdoor_overall", "trustpilot_score"]:
        panel[f"{v}_bar"] = panel.groupby("ticker")[v].transform("mean")

    # --- Merge Diagnostics ---
    print("\n  MERGE DIAGNOSTICS:")
    print(f"  {'Variable':<25s} {'Non-null':>8s} {'% Complete':>10s} {'Firms':>6s}")
    print("  " + "-"*55)
    for var in ["revenue", "capex_intensity", "trustpilot_score",
                "glassdoor_overall", "ecom_share"]:
        nn = panel[var].notna().sum()
        pct = 100 * nn / len(panel)
        firms = panel[panel[var].notna()].ticker.nunique()
        print(f"  {var:<25s} {nn:>8d} {pct:>9.1f}% {firms:>6d}")

    # Complete observations (all three: T + W + K)
    complete = panel.dropna(subset=["trustpilot_score", "glassdoor_overall",
                                     "capex_intensity"])
    n_complete = len(complete)
    print(f"\n  Complete observations (T + W + K): {n_complete} "
          f"({100*n_complete/max(len(panel),1):.0f}%)")
    print(f"  Firms in complete panel: {complete.ticker.nunique()}")
    print(f"  Years: {sorted(complete.year.unique())}")

    # Balance check
    print("\n  BALANCE CHECK (complete obs by firm × year):")
    pivot = complete.pivot_table(index="ticker", columns="year",
                                values="trustpilot_score", aggfunc="count")
    print("  " + pivot.fillna(".").to_string().replace("\n", "\n  "))

    # Save
    outpath = "data/processed/eu_empirical_panel.csv"
    os.makedirs("data/processed", exist_ok=True)
    panel.to_csv(outpath, index=False)
    print(f"\n  Saved {len(panel)} obs to {outpath}")

    return panel


# -----------------------------------------------------------------------
# 2. Summary Statistics
# -----------------------------------------------------------------------

def summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("PHASE 2: SUMMARY STATISTICS")
    print("="*70)

    vars_desc = [
        ("trustpilot_score", "Trustpilot Score (T proxy)"),
        ("glassdoor_overall", "Glassdoor Overall (W proxy)"),
        ("capex_intensity", "Capex/Revenue (K proxy)"),
        ("sga_intensity", "SGA/Revenue"),
        ("ecom_share", "E-commerce Share %"),
        ("log_revenue", "Log Revenue"),
    ]

    print(f"\n  {'Variable':<30s} {'N':>5s} {'Mean':>8s} {'SD':>8s} "
          f"{'Min':>8s} {'Max':>8s}")
    print("  " + "-"*65)
    for var, label in vars_desc:
        s = df[var].dropna()
        if len(s) > 0:
            print(f"  {label:<30s} {len(s):>5d} {s.mean():>8.3f} "
                  f"{s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f}")

    # Cross-sectional correlations
    print("\n  CROSS-SECTIONAL CORRELATIONS (firm means):")
    means = df.groupby("ticker")[["trustpilot_score", "glassdoor_overall",
                                   "capex_intensity"]].mean().dropna()
    print(f"  Corr(Trustpilot, Glassdoor):    {means['trustpilot_score'].corr(means['glassdoor_overall']):.3f}")
    print(f"  Corr(Trustpilot, Capex_int):    {means['trustpilot_score'].corr(means['capex_intensity']):.3f}")
    print(f"  Corr(Glassdoor, Capex_int):     {means['glassdoor_overall'].corr(means['capex_intensity']):.3f}")

    # Within-firm variation
    print("\n  WITHIN-FIRM VARIATION:")
    for var in ["trustpilot_score", "glassdoor_overall", "capex_intensity"]:
        demeaned = df.groupby("ticker")[var].transform(lambda x: x - x.mean())
        overall_sd = df[var].std()
        within_sd = demeaned.std()
        ratio = within_sd / max(overall_sd, 0.001)
        print(f"  {var:<25s}  within_SD={within_sd:.4f}  "
              f"overall_SD={overall_sd:.4f}  ratio={ratio:.2f}")


# -----------------------------------------------------------------------
# 3. Regression helpers
# -----------------------------------------------------------------------

def sig(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""


def run(df, y, xs, method="fe", label="", cluster="ticker"):
    """Run regression with specified method."""
    cols = [y] + xs + [cluster, "year"]
    sub = df[cols].dropna()
    n = len(sub); nf = sub[cluster].nunique()
    if n < 15 or nf < 4:
        print(f"  [{label}] N={n}, firms={nf} — too few")
        return None
    panel = sub.set_index([cluster, "year"])
    Y = panel[y]; X = panel[xs]

    try:
        if method == "fe":
            mod = PanelOLS(Y, X, entity_effects=True, time_effects=True,
                           check_rank=False, drop_absorbed=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            r2l = f"R2w={res.rsquared_within:.3f}"
        elif method == "be":
            mod = BetweenOLS(Y, sm.add_constant(X))
            res = mod.fit()
            r2l = f"R2b={res.rsquared:.3f}"
        elif method == "re":
            mod = RandomEffects(Y, sm.add_constant(X))
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            r2l = f"R2={res.rsquared:.3f}"
        else:
            Xc = sm.add_constant(X)
            grp = sub[cluster].astype("category").cat.codes.values
            res = sm.OLS(Y, Xc).fit(cov_type="cluster",
                                     cov_kwds={"groups": grp})
            r2l = f"R2={res.rsquared:.3f}"
    except Exception as e:
        print(f"  [{label}] Error: {e}")
        return None

    print(f"\n  [{label}] N={n}, firms={nf}, {r2l}")
    result = {}
    for v in xs:
        if v in res.params.index:
            b = res.params[v]
            se_ = res.std_errors[v] if hasattr(res, 'std_errors') else res.bse[v]
            t_ = res.tstats[v] if hasattr(res, 'tstats') else res.tvalues[v]
            p_ = res.pvalues[v]
            marker = " <<<" if sig(p_) and "_x_" in v.lower() else ""
            print(f"    {v:40s}  b={b:>10.4f}  se={se_:>9.4f}  "
                  f"t={t_:>7.3f}  p={p_:.4f} {sig(p_)}{marker}")
            result[v] = {"b": b, "se": se_, "t": t_, "p": p_}
    return result


# -----------------------------------------------------------------------
# 4. Core Tests (Replication of US Tests 1-3)
# -----------------------------------------------------------------------

def run_core_tests(df):
    """Run the three core WCT tests on European data."""
    print("\n" + "="*70)
    print("PHASE 3: CORE REPLICATION TESTS")
    print("="*70)

    # --- Test 1: Complementarity (K×W) ---
    # Use Trustpilot as T (outcome), Glassdoor as W, Capex as K
    print("\n" + "-"*60)
    print("TEST 1: COMPLEMENTARITY (K×W → Trustpilot)")
    print("-"*60)
    print("  Note: Trustpilot (1-5) replaces ACSI (0-100) as T proxy.")

    run(df, "trustpilot_score",
        ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z", "log_revenue"],
        "be", "T1a: Between-effects (main)")

    run(df, "trustpilot_score",
        ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z", "log_revenue"],
        "fe", "T1b: Two-way FE")

    run(df, "trustpilot_score",
        ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z", "log_revenue"],
        "pooled", "T1c: Pooled OLS")

    # --- Test 2: Front-Stage Trap (inverted-U in e-commerce) ---
    print("\n" + "-"*60)
    print("TEST 2: FRONT-STAGE TRAP (e-commerce → Trustpilot)")
    print("-"*60)

    run(df, "trustpilot_score",
        ["ecom_share", "ecom2", "log_revenue"],
        "fe", "T2a: Quadratic ecom (FE)")

    run(df, "trustpilot_score",
        ["ecom_share", "ecom2", "log_revenue"],
        "be", "T2b: Quadratic ecom (Between)")

    run(df, "trustpilot_score",
        ["ecom_share", "ecom2", "glassdoor_overall", "log_revenue"],
        "fe", "T2c: Quadratic ecom + W (FE)")

    # --- Test 3: Good Jobs Buffer (ΔW × K → ΔT) ---
    print("\n" + "-"*60)
    print("TEST 3: GOOD JOBS BUFFER (ΔGlassdoor × K → ΔTrustpilot)")
    print("-"*60)

    df_s = df.sort_values(["ticker", "year"]).copy()
    df_s["d_trust"] = df_s.groupby("ticker")["trustpilot_score"].diff()
    df_s["d_gd"] = df_s.groupby("ticker")["glassdoor_overall"].diff()
    df_s["d_gd_x_K"] = df_s["d_gd"] * df_s["capex_intensity"]

    run(df_s, "d_trust", ["d_gd", "d_gd_x_K", "capex_intensity"],
        "fe", "T3a: ΔTrust ~ ΔGD×K (FE)")


# -----------------------------------------------------------------------
# 5. Robustness Battery
# -----------------------------------------------------------------------

def run_robustness(df):
    """Run robustness checks matching US analysis."""
    print("\n" + "="*70)
    print("PHASE 4: ROBUSTNESS BATTERY")
    print("="*70)

    xs = ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z", "log_revenue"]

    # R1: Lagged specification
    print("\n--- R1: Lagged K×W (t-1 → t) ---")
    run(df, "trustpilot_score",
        ["capex_intensity_z_L1", "glassdoor_overall_z_L1",
         "K_x_W_z_L1", "log_revenue"],
        "be", "R1a: Lagged BE")

    # R2: Exclude COVID
    print("\n--- R2: Exclude 2020-2021 ---")
    df_nc = df[~df.year.isin([2020, 2021])]
    run(df_nc, "trustpilot_score", xs, "be", "R2a: Excl COVID (BE)")

    # R3: UK-only sub-sample
    print("\n--- R3: UK-only sub-sample ---")
    df_uk = df[df.country == "GB"]
    run(df_uk, "trustpilot_score", xs, "be", "R3a: UK only (BE)")
    run(df_uk, "trustpilot_score", xs, "fe", "R3b: UK only (FE)")

    # R4: Continental Europe only
    print("\n--- R4: Continental Europe only ---")
    df_eu = df[df.country != "GB"]
    run(df_eu, "trustpilot_score", xs, "be", "R4a: Continental EU (BE)")

    # R5: Leave-one-out
    print("\n--- R5: Leave-one-out (Between-effects) ---")
    core = df.dropna(subset=["trustpilot_score"] + xs + ["ticker", "year"])
    firms = sorted(core.ticker.unique())
    print(f"  {'Dropped':>12s}  {'b(K×W)':>10s}  {'p':>8s}")
    print("  " + "-"*35)

    for firm in firms:
        df_loo = df[df.ticker != firm].copy()
        for v in ["capex_intensity", "glassdoor_overall"]:
            mu, sd = df_loo[v].mean(), df_loo[v].std()
            if sd > 0:
                df_loo[f"{v}_z"] = (df_loo[v] - mu) / sd
        df_loo["K_x_W_z"] = df_loo["capex_intensity_z"] * df_loo["glassdoor_overall_z"]

        sub = df_loo.dropna(subset=["trustpilot_score"] + xs + ["ticker", "year"])
        if sub.ticker.nunique() < 4:
            continue
        panel = sub.set_index(["ticker", "year"])
        Y = panel["trustpilot_score"]
        X = sm.add_constant(panel[xs])
        try:
            res = BetweenOLS(Y, X).fit()
            b = res.params.get("K_x_W_z", np.nan)
            p = res.pvalues.get("K_x_W_z", np.nan)
            name = EU_RETAILERS.get(firm, (firm,))[0] if firm in EU_RETAILERS else firm
            print(f"  {name:>12s}  {b:>10.4f}  {p:>8.4f} {sig(p)}")
        except Exception:
            pass

    # R6: Wild cluster bootstrap
    print("\n--- R6: Wild cluster bootstrap ---")
    firm_means = core.groupby("ticker").agg({
        "trustpilot_score": "mean", "capex_intensity_z": "mean",
        "glassdoor_overall_z": "mean", "K_x_W_z": "mean",
        "log_revenue": "mean",
    }).reset_index()

    Y = firm_means["trustpilot_score"].values
    X = sm.add_constant(firm_means[xs].values)
    ols = sm.OLS(Y, X).fit()
    b_kxw = ols.params[3]
    resid = ols.resid

    np.random.seed(42)
    n_boot = 9999
    boot_coefs = np.zeros(n_boot)
    for i in range(n_boot):
        weights = np.random.choice([-1, 1], size=len(firm_means))
        Y_boot = ols.fittedvalues + resid * weights
        try:
            boot_coefs[i] = sm.OLS(Y_boot, X).fit().params[3]
        except Exception:
            boot_coefs[i] = np.nan

    boot_coefs = boot_coefs[~np.isnan(boot_coefs)]
    p_boot = np.mean(np.abs(boot_coefs - np.mean(boot_coefs)) >= np.abs(b_kxw))
    ci_lo = np.percentile(boot_coefs, 2.5)
    ci_hi = np.percentile(boot_coefs, 97.5)

    print(f"  K×W point estimate: {b_kxw:.4f}")
    print(f"  Bootstrap p-value:  {p_boot:.4f} {sig(p_boot)}")
    print(f"  Bootstrap 95% CI:   [{ci_lo:.4f}, {ci_hi:.4f}]")

    # R7: Placebo test
    print("\n--- R7: Placebo test (randomize K×W) ---")
    np.random.seed(42)
    kxw_orig = firm_means["K_x_W_z"].values.copy()
    placebo_coefs = []
    for _ in range(1000):
        np.random.shuffle(kxw_orig)
        X_p = X.copy()
        X_p[:, 3] = kxw_orig
        try:
            placebo_coefs.append(sm.OLS(Y, X_p).fit().params[3])
        except Exception:
            pass
    p_placebo = np.mean(np.array(placebo_coefs) >= b_kxw)
    print(f"  Actual K×W: {b_kxw:.4f}")
    print(f"  Placebo mean: {np.mean(placebo_coefs):.4f}")
    print(f"  Placebo p-value: {p_placebo:.4f} {sig(p_placebo)}")

    # R8: Macro controls
    print("\n--- R8: With macro controls ---")
    run(df, "trustpilot_score",
        xs + ["ecom_share", "time_trend"],
        "pooled", "R8a: + ecom + trend (Pooled)")


# -----------------------------------------------------------------------
# 6. US vs EU Comparison
# -----------------------------------------------------------------------

def compare_us_eu(eu_df):
    """Compare key results between US and EU samples."""
    print("\n" + "="*70)
    print("PHASE 5: US vs EU COMPARISON")
    print("="*70)

    # Load US results if available
    us_path = "data/processed/regression_results.csv"
    if os.path.exists(us_path):
        us_res = pd.read_csv(us_path)
        # Find complementarity coefficient from between-effects
        comp = us_res[(us_res["test"] == "complementarity") &
                      (us_res["variable"] == "K_x_W_z")]
        if len(comp) > 0:
            for _, row in comp.iterrows():
                print(f"\n  US {row['model']}: b={row['coefficient']:.4f}, "
                      f"p={row['p_value']:.4f}")
    else:
        print("\n  [US results not found — run 06_regressions.py first]")

    # EU between-effects
    xs = ["capex_intensity_z", "glassdoor_overall_z", "K_x_W_z", "log_revenue"]
    core = eu_df.dropna(subset=["trustpilot_score"] + xs + ["ticker", "year"])
    firm_means = core.groupby("ticker").agg({
        "trustpilot_score": "mean", **{v: "mean" for v in xs}
    }).reset_index()

    Y = firm_means["trustpilot_score"].values
    X = sm.add_constant(firm_means[xs].values)
    res = sm.OLS(Y, X).fit()

    print(f"\n  EU Between-effects: b(K×W)={res.params[3]:.4f}, "
          f"p={res.pvalues[3]:.4f}")

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  CROSS-GEOGRAPHY COMPARISON                             │
  ├──────────────┬───────────────┬───────────────────────────┤
  │              │  US (ACSI)    │  EU (Trustpilot)          │
  ├──────────────┼───────────────┼───────────────────────────┤
  │  T proxy     │  ACSI 0-100   │  Trustpilot 1-5           │
  │  W proxy     │  Glassdoor    │  Glassdoor/Kununu         │
  │  K proxy     │  Capex/Rev    │  Capex/Rev (yfinance)     │
  │  N firms     │  13           │  {core.ticker.nunique():>2d}                        │
  │  K×W sign    │  Positive     │  {'Positive' if res.params[3]>0 else 'Negative'}                   │
  │  K×W p-value │  0.003        │  {res.pvalues[3]:.3f}                     │
  └──────────────┴───────────────┴───────────────────────────┘
    """)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    df = assemble_panel()
    summary_stats(df)
    run_core_tests(df)
    run_robustness(df)
    compare_us_eu(df)

    print("\n" + "="*70)
    print("EUROPEAN REPLICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
