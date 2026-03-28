#!/usr/bin/env python3
"""
06_regressions.py — Run the three core empirical tests from the WCT model.

Tests:
  1. Complementarity (p > 0):
     ACSI_it = alpha_i + gamma_t + b1*Capex_it + b2*Glassdoor_it
               + b3*(Capex x Glassdoor)_it + X_it*delta + e_it
     Prediction: b3 > 0

  2. Front-Stage Trap (inverted-U):
     ACSI_it = alpha_i + gamma_t + b1*EcomShare_it + b2*EcomShare^2_it
               + X_it*delta + e_it
     Prediction: b1 > 0, b2 < 0

  3. Good Jobs Buffer:
     dACSI_it = alpha_i + gamma_t + b1*dGlassdoor_it
                + b2*(dGlassdoor x Capex)_it + X_it*delta + e_it
     Prediction: b2 > 0

Requires: pandas, statsmodels, linearmodels (for fixed effects)
Fallback: If linearmodels not available, uses statsmodels OLS with dummies.

Output:
  - Console: regression tables
  - data/processed/regression_results.csv: coefficient estimates
  - data/processed/regression_tables.tex: LaTeX-formatted tables

Usage:
  python 06_regressions.py
"""

import csv
import math
import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import (
        cov_cluster as cluster_cov,
    )
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    from linearmodels.panel import PanelOLS
    HAS_LM = True
except ImportError:
    HAS_LM = False

try:
    from scipy.stats import t as t_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_panel(filepath: str = "data/processed/wct_empirical_panel.csv"):
    """Load the assembled panel into a pandas DataFrame."""
    if not HAS_PANDAS:
        print("[ERROR] pandas is required. Install: pip install pandas")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"[REG] Loaded panel: {len(df)} obs, {df['ticker'].nunique()} firms, "
          f"years {df['year'].min()}-{df['year'].max()}")

    # Convert numeric columns
    numeric_cols = [
        "acsi_score", "glassdoor_overall", "glassdoor_worklife",
        "capex_intensity", "sga_intensity", "net_margin",
        "log_revenue", "log_assets", "log_employees",
        "ecom_share_pct", "K_x_W", "revenue", "capex",
        "retail_turnover_rate", "retail_avg_wage",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def standardise(series: pd.Series) -> pd.Series:
    """Standardise a series to mean 0, SD 1."""
    return (series - series.mean()) / series.std()


def run_ols_with_fe(df: pd.DataFrame, y_col: str, x_cols: list,
                    firm_fe: bool = True, year_fe: bool = True,
                    cluster_col: str = "ticker",
                    label: str = "Model") -> dict:
    """
    Run OLS regression with firm and/or year fixed effects.

    Uses linearmodels PanelOLS if available; otherwise dummies via statsmodels.

    Small-sample corrections (applied when clustering):
      - p-values recomputed using t(G-1) distribution where G = number of
        clusters, following Cameron, Gelbach & Miller (2008)
      - Reported alongside standard inference for transparency

    Returns a dict of results.
    """
    # Drop rows with missing values in relevant columns
    cols_needed = [y_col] + x_cols + [cluster_col, "year"]
    subset = df[cols_needed].dropna()
    n = len(subset)
    n_firms = subset[cluster_col].nunique()
    if n < 30:
        print(f"  [{label}] Only {n} complete observations — skipping.")
        return {"label": label, "n": n, "status": "insufficient_data"}

    # Determine effective df for p-value correction
    # With G clusters, use t(G-1) instead of t(N-k) or z
    uses_cluster = firm_fe  # we cluster whenever we have firm FE
    df_eff = n_firms - 1 if uses_cluster and HAS_SCIPY else None

    fe_note = ""
    if df_eff is not None:
        fe_note = f", p-vals: t({df_eff})"

    print(f"  [{label}] N={n}, firms={n_firms}, "
          f"years={subset['year'].nunique()}{fe_note}")

    if HAS_LM and firm_fe:
        # Use linearmodels PanelOLS for proper within-estimator
        panel_df = subset.set_index([cluster_col, "year"])
        y = panel_df[y_col]
        X = panel_df[x_cols]
        X = sm.add_constant(X) if not firm_fe else X  # no constant with FE

        model = PanelOLS(y, X, entity_effects=firm_fe,
                         time_effects=year_fe, check_rank=False)
        res = model.fit(cov_type="clustered", cluster_entity=True)

        result = {
            "label": label,
            "n": n,
            "n_firms": n_firms,
            "df_eff": df_eff,
            "r2_within": round(res.rsquared_within, 4),
            "r2_overall": round(res.rsquared_overall, 4) if hasattr(res, "rsquared_overall") else "",
            "status": "ok",
            "coefficients": {},
        }
        for var in x_cols:
            if var in res.params.index:
                t_val = res.tstats[var]
                # Recompute p-value using t(G-1)
                if df_eff is not None and df_eff > 0:
                    p_corrected = 2 * (1 - t_dist.cdf(abs(t_val), df_eff))
                else:
                    p_corrected = res.pvalues[var]
                result["coefficients"][var] = {
                    "coef": round(res.params[var], 6),
                    "se": round(res.std_errors[var], 6),
                    "t": round(t_val, 3),
                    "p": round(float(p_corrected), 4),
                    "p_raw": round(float(res.pvalues[var]), 4),
                    "sig": _sig_stars(float(p_corrected)),
                }
        return result

    elif HAS_SM:
        # Fallback: OLS with dummy variables
        y = subset[y_col]
        X = subset[x_cols].copy()

        if firm_fe:
            firm_dummies = pd.get_dummies(subset[cluster_col],
                                          prefix="firm", drop_first=True,
                                          dtype=float)
            X = pd.concat([X, firm_dummies], axis=1)

        if year_fe:
            year_dummies = pd.get_dummies(subset["year"],
                                          prefix="yr", drop_first=True,
                                          dtype=float)
            X = pd.concat([X, year_dummies], axis=1)

        X = sm.add_constant(X)
        model = sm.OLS(y, X)

        # Cluster standard errors at firm level
        groups = subset[cluster_col].astype("category").cat.codes
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": groups})

        result = {
            "label": label,
            "n": n,
            "n_firms": n_firms,
            "df_eff": df_eff,
            "r2": round(res.rsquared, 4),
            "r2_adj": round(res.rsquared_adj, 4),
            "status": "ok",
            "coefficients": {},
        }
        for var in x_cols:
            if var in res.params.index:
                t_val = res.tvalues[var]
                if df_eff is not None and df_eff > 0:
                    p_corrected = 2 * (1 - t_dist.cdf(abs(t_val), df_eff))
                else:
                    p_corrected = res.pvalues[var]
                result["coefficients"][var] = {
                    "coef": round(res.params[var], 6),
                    "se": round(res.bse[var], 6),
                    "t": round(float(t_val), 3),
                    "p": round(float(p_corrected), 4),
                    "p_raw": round(float(res.pvalues[var]), 4),
                    "sig": _sig_stars(float(p_corrected)),
                }
        return result

    else:
        print("[ERROR] Neither linearmodels nor statsmodels available.")
        return {"label": label, "n": n, "status": "no_library"}


def _sig_stars(p: float) -> str:
    """Return significance stars."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


# -----------------------------------------------------------------------
# Test 1: Complementarity
# -----------------------------------------------------------------------
def test_complementarity(df: pd.DataFrame) -> dict:
    """
    Test 1: Is the K × W interaction positive?
    H0: b3 <= 0 (no complementarity)
    H1: b3 > 0  (workforce-technology complementarity)
    """
    print("\n" + "="*60)
    print("TEST 1: COMPLEMENTARITY (p > 0)")
    print("="*60)

    # Standardise key variables for interpretability
    df = df.copy()
    df["capex_z"] = standardise(df["capex_intensity"])
    df["gd_z"] = standardise(df["glassdoor_overall"])
    df["K_x_W_z"] = df["capex_z"] * df["gd_z"]

    # Model 1a: Basic (no FE)
    res_a = run_ols_with_fe(
        df, y_col="acsi_score",
        x_cols=["capex_z", "gd_z", "K_x_W_z", "log_revenue"],
        firm_fe=False, year_fe=False,
        label="1a: OLS (no FE)",
    )

    # Model 1b: Firm FE only
    res_b = run_ols_with_fe(
        df, y_col="acsi_score",
        x_cols=["capex_z", "gd_z", "K_x_W_z", "log_revenue"],
        firm_fe=True, year_fe=False,
        label="1b: Firm FE",
    )

    # Model 1c: Two-way FE (main specification)
    res_c = run_ols_with_fe(
        df, y_col="acsi_score",
        x_cols=["capex_z", "gd_z", "K_x_W_z", "log_revenue"],
        firm_fe=True, year_fe=True,
        label="1c: Two-way FE (main)",
    )

    # Model 1d: With additional controls
    res_d = run_ols_with_fe(
        df, y_col="acsi_score",
        x_cols=["capex_z", "gd_z", "K_x_W_z", "log_revenue",
                "sga_intensity", "net_margin"],
        firm_fe=True, year_fe=True,
        label="1d: Two-way FE + controls",
    )

    results = [res_a, res_b, res_c, res_d]
    _print_results_table(results, "K_x_W_z", "Complementarity (p)")
    return {"test": "complementarity", "results": results}


# -----------------------------------------------------------------------
# Test 2: Front-Stage Trap
# -----------------------------------------------------------------------
def test_front_stage_trap(df: pd.DataFrame) -> dict:
    """
    Test 2: Is there an inverted-U between digitalisation and ACSI?
    H0: b2 >= 0 (monotonic positive or no relationship)
    H1: b2 < 0  (inverted-U: digitalisation helps then hurts)
    """
    print("\n" + "="*60)
    print("TEST 2: FRONT-STAGE TRAP (inverted-U)")
    print("="*60)

    df = df.copy()
    df["ecom2"] = df["ecom_share_pct"] ** 2

    # Model 2a: Linear (benchmark)
    res_a = run_ols_with_fe(
        df, y_col="acsi_score",
        x_cols=["ecom_share_pct", "log_revenue"],
        firm_fe=True, year_fe=False,
        label="2a: Linear (firm FE)",
    )

    # Model 2b: Quadratic (main specification)
    res_b = run_ols_with_fe(
        df, y_col="acsi_score",
        x_cols=["ecom_share_pct", "ecom2", "log_revenue"],
        firm_fe=True, year_fe=False,
        label="2b: Quadratic (firm FE)",
    )

    # Model 2c: With workforce control
    res_c = run_ols_with_fe(
        df, y_col="acsi_score",
        x_cols=["ecom_share_pct", "ecom2", "glassdoor_overall",
                "log_revenue"],
        firm_fe=True, year_fe=False,
        label="2c: Quad + Glassdoor",
    )

    results = [res_a, res_b, res_c]
    _print_results_table(results, "ecom2", "Front-Stage Trap (ecom^2)")

    # Compute turning point if quadratic is significant
    for r in results:
        if r.get("status") == "ok":
            coeffs = r.get("coefficients", {})
            b1 = coeffs.get("ecom_share_pct", {}).get("coef")
            b2 = coeffs.get("ecom2", {}).get("coef")
            if b1 and b2 and b2 < 0:
                turning = -b1 / (2 * b2)
                print(f"  [{r['label']}] Turning point: "
                      f"e-commerce share = {turning:.1f}%")

    return {"test": "front_stage_trap", "results": results}


# -----------------------------------------------------------------------
# Test 3: Good Jobs Buffer
# -----------------------------------------------------------------------
def test_good_jobs_buffer(df: pd.DataFrame) -> dict:
    """
    Test 3: Does workforce improvement matter more when tech base is large?
    H0: b2 <= 0
    H1: b2 > 0  (Good Jobs Buffer)
    """
    print("\n" + "="*60)
    print("TEST 3: GOOD JOBS BUFFER")
    print("="*60)

    df = df.copy()
    df = df.sort_values(["ticker", "year"])

    # Compute first differences within firm
    df["d_acsi"] = df.groupby("ticker")["acsi_score"].diff()
    df["d_glassdoor"] = df.groupby("ticker")["glassdoor_overall"].diff()
    df["d_gd_x_capex"] = df["d_glassdoor"] * df["capex_intensity"]

    # Model 3a: Simple first-difference
    res_a = run_ols_with_fe(
        df, y_col="d_acsi",
        x_cols=["d_glassdoor"],
        firm_fe=True, year_fe=True,
        label="3a: dACSI on dGlassdoor",
    )

    # Model 3b: With interaction (main specification)
    res_b = run_ols_with_fe(
        df, y_col="d_acsi",
        x_cols=["d_glassdoor", "d_gd_x_capex", "capex_intensity"],
        firm_fe=True, year_fe=True,
        label="3b: + dGD × Capex (main)",
    )

    # Model 3c: With controls
    res_c = run_ols_with_fe(
        df, y_col="d_acsi",
        x_cols=["d_glassdoor", "d_gd_x_capex", "capex_intensity",
                "log_revenue", "sga_intensity"],
        firm_fe=True, year_fe=True,
        label="3c: + controls",
    )

    results = [res_a, res_b, res_c]
    _print_results_table(results, "d_gd_x_capex", "Good Jobs Buffer")
    return {"test": "good_jobs_buffer", "results": results}


# -----------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------

def _print_results_table(results: list[dict], key_var: str, title: str):
    """Print a formatted regression results table."""
    print(f"\n--- {title} ---")
    print(f"{'Model':<35} {'b':>10} {'SE':>10} {'t':>8} "
          f"{'p[t(G-1)]':>10} {'Sig':>5} {'N':>6}")
    print("-" * 92)
    for r in results:
        if r.get("status") != "ok":
            print(f"{r.get('label', '?'):<35} {'—':>10} {'':>10} {'':>8} "
                  f"{'':>10} {'':>5} {r.get('n', 0):>6}  [{r.get('status')}]")
            continue
        coeff = r.get("coefficients", {}).get(key_var, {})
        if coeff:
            p_raw = coeff.get("p_raw", coeff["p"])
            p_str = f"{coeff['p']:.4f}"
            if p_raw != coeff["p"]:
                p_str += f"({p_raw:.3f})"
            print(f"{r['label']:<35} {coeff['coef']:>10.5f} "
                  f"{coeff['se']:>10.5f} {coeff['t']:>8.3f} "
                  f"{p_str:>10} {coeff['sig']:>5} {r['n']:>6}")
        else:
            print(f"{r['label']:<35} {'n/a':>10} {'':>10} {'':>8} "
                  f"{'':>10} {'':>5} {r['n']:>6}")
    print()


def save_results_csv(all_results: list[dict],
                     outpath: str = "data/processed/regression_results.csv"):
    """Save all regression coefficients to CSV."""
    rows = []
    for test_result in all_results:
        test_name = test_result.get("test", "")
        for model in test_result.get("results", []):
            if model.get("status") != "ok":
                continue
            for var, coeff in model.get("coefficients", {}).items():
                rows.append({
                    "test": test_name,
                    "model": model["label"],
                    "variable": var,
                    "coefficient": coeff["coef"],
                    "std_error": coeff["se"],
                    "t_stat": coeff["t"],
                    "p_value": coeff["p"],
                    "significance": coeff["sig"],
                    "n": model["n"],
                })

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test", "model", "variable", "coefficient",
            "std_error", "t_stat", "p_value", "significance", "n",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[REG] Saved {len(rows)} coefficient estimates to {outpath}")


def save_latex_table(all_results: list[dict],
                     outpath: str = "data/processed/regression_tables.tex"):
    """Generate a LaTeX regression table for the paper."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Empirical Tests of Model Predictions}")
    lines.append(r"\label{tab:empirical}")
    lines.append(r"\small")

    for test_result in all_results:
        test_name = test_result.get("test", "").replace("_", " ").title()
        models = [m for m in test_result.get("results", [])
                  if m.get("status") == "ok"]
        if not models:
            continue

        # Collect all variables across models
        all_vars = []
        for m in models:
            for v in m.get("coefficients", {}).keys():
                if v not in all_vars:
                    all_vars.append(v)

        n_models = len(models)
        col_spec = "l" + "c" * n_models
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")

        # Header
        headers = " & ".join([m["label"].split(":")[0] for m in models])
        lines.append(f"& {headers} \\\\")
        lines.append(r"\midrule")

        # Coefficient rows
        for var in all_vars:
            var_label = var.replace("_", r"\_")
            cells = []
            for m in models:
                c = m.get("coefficients", {}).get(var)
                if c:
                    cells.append(f"{c['coef']:.4f}{c['sig']}")
                else:
                    cells.append("")
            lines.append(f"{var_label} & " + " & ".join(cells) + r" \\")

            # Standard errors row
            se_cells = []
            for m in models:
                c = m.get("coefficients", {}).get(var)
                if c:
                    se_cells.append(f"({c['se']:.4f})")
                else:
                    se_cells.append("")
            lines.append(" & " + " & ".join(se_cells) + r" \\")

        lines.append(r"\midrule")
        # N row
        n_cells = [str(m["n"]) for m in models]
        lines.append(r"$N$ & " + " & ".join(n_cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

    lines.append(r"\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"[REG] Saved LaTeX table to {outpath}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    panel_path = "data/processed/wct_empirical_panel.csv"
    if not os.path.exists(panel_path):
        print(f"[ERROR] Panel not found: {panel_path}")
        print("Run 05_assemble_panel.py first.")
        sys.exit(1)

    df = load_panel(panel_path)

    # Descriptive statistics
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    desc_vars = ["acsi_score", "glassdoor_overall", "capex_intensity",
                 "sga_intensity", "ecom_share_pct", "log_revenue"]
    for var in desc_vars:
        if var in df.columns:
            s = df[var].dropna()
            if len(s) > 0:
                print(f"  {var:<25} N={len(s):>4}  "
                      f"mean={s.mean():.3f}  sd={s.std():.3f}  "
                      f"min={s.min():.3f}  max={s.max():.3f}")

    # Run tests
    all_results = []
    all_results.append(test_complementarity(df))
    all_results.append(test_front_stage_trap(df))
    all_results.append(test_good_jobs_buffer(df))

    # Save outputs
    save_results_csv(all_results)
    save_latex_table(all_results)

    # Summary verdict
    print("\n" + "="*60)
    print("VERDICT: IS p DETECTABLE?")
    print("="*60)
    comp = all_results[0]
    main_model = None
    for m in comp.get("results", []):
        if "main" in m.get("label", "").lower():
            main_model = m
            break
    if main_model and main_model.get("status") == "ok":
        kw = main_model["coefficients"].get("K_x_W_z", {})
        if kw:
            p_val = kw.get("p", 1)
            coef = kw.get("coef", 0)
            if p_val < 0.05 and coef > 0:
                print(f"  YES — b3 = {coef:.4f} (p = {p_val:.4f})")
                print("  → Pursue Management Science with full empirical section")
            elif p_val < 0.10 and coef > 0:
                print(f"  MARGINAL — b3 = {coef:.4f} (p = {p_val:.4f})")
                print("  → Pursue MSOM with suggestive evidence + calibration")
            else:
                print(f"  NOT DETECTED — b3 = {coef:.4f} (p = {p_val:.4f})")
                print("  → Submit to Service Science as theory paper; "
                      "note data limitations")
        else:
            print("  [WARN] K_x_W_z coefficient not found in main model")
    else:
        print("  [WARN] Main complementarity model did not converge")

    print("\n[DONE] Full regression output saved to data/processed/")


if __name__ == "__main__":
    main()
