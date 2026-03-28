#!/usr/bin/env python3
"""
11_county_regressions.py — County-level Good Jobs Buffer regressions.

Tests whether counties with higher initial retail wages show more resilient
retail employment as national e-commerce penetration grows.

Data: BLS Quarterly Census of Employment and Wages (QCEW), NAICS 44-45,
      2015-2024, county-level annual averages.

Specification (Eq. 14 in main paper):
  ln(E_ct) - ln(E_c,2015) = alpha_c + gamma_t + beta_1 (W_z_c,2015 × Ecom_t) + eps_ct

Because Ecom_t is a national-level variable (identical across counties within
each year), its main effect is perfectly collinear with year FE and absorbed
by them. W_c,2015 is time-invariant and absorbed by county FE. beta_1 thus
identifies the differential effect of e-commerce growth on counties with
higher initial retail wages.

Output: printed regression tables + data/processed/county_regression_results.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t as t_dist
from linearmodels.panel import PanelOLS
import os
from pathlib import Path

# ── National e-commerce share (FRED ECOMPCTSA, annual averages) ──
# Source: Federal Reserve Economic Data, series ECOMPCTSA
# https://fred.stlouisfed.org/series/ECOMPCTSA
ECOM_SHARE = {
    2015:  7.4,
    2016:  8.1,
    2017:  9.0,
    2018:  9.9,
    2019: 10.7,
    2020: 14.0,
    2021: 13.2,
    2022: 14.5,
    2023: 15.4,
    2024: 16.1,
}


def load_and_filter(project_root: Path) -> pd.DataFrame:
    """Load BLS county panel and apply documented filters."""
    df = pd.read_csv(project_root / 'data' / 'bls_qcew' / 'retail_county_panel.csv')
    print(f"Raw data: {len(df)} rows, {df['fips'].nunique()} counties, "
          f"years {df['year'].min()}-{df['year'].max()}")

    # Baseline filters (applied on 2015 values)
    baseline = df[df['year'] == 2015].copy()
    baseline_pass = baseline[
        (baseline['annual_avg_emplvl'] > 50) &
        (baseline['annual_avg_wkly_wage'] > 200)
    ]
    good_fips = baseline_pass['fips'].unique()
    print(f"Counties passing baseline filter (empl > 50, wage > $200): {len(good_fips)}")

    # Keep only passing counties, drop disclosure-suppressed rows
    panel = df[df['fips'].isin(good_fips)].copy()
    panel = panel[(panel['annual_avg_emplvl'] > 0) & (panel['annual_avg_wkly_wage'] > 0)]
    print(f"Analysis panel: {len(panel)} county-year obs, {panel['fips'].nunique()} counties")

    return panel, baseline_pass


def build_regression_variables(panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Construct regression variables."""
    # Baseline (2015) wage — standardised cross-sectionally
    wage_2015 = baseline.set_index('fips')['annual_avg_wkly_wage']
    w_mean = wage_2015.mean()
    w_std = wage_2015.std()
    print(f"\nBaseline wage: mean=${w_mean:.0f}/week, SD=${w_std:.0f}/week")

    # Merge baseline wage
    panel = panel.copy()
    panel['wage_2015'] = panel['fips'].map(wage_2015)
    panel['wage_2015_z'] = (panel['wage_2015'] - w_mean) / w_std

    # Baseline employment (for log growth dependent variable)
    empl_2015 = baseline.set_index('fips')['annual_avg_emplvl']
    panel['empl_2015'] = panel['fips'].map(empl_2015)

    # Dependent variable: log employment growth relative to 2015
    panel['ln_empl_growth'] = np.log(panel['annual_avg_emplvl']) - np.log(panel['empl_2015'])

    # E-commerce share (national, time-varying)
    panel['ecom'] = panel['year'].map(ECOM_SHARE)

    # Interaction: wage_z × ecom
    panel['wage_z_x_ecom'] = panel['wage_2015_z'] * panel['ecom']

    # Log wage growth (for mechanism check)
    wage_map_2015 = baseline.set_index('fips')['annual_avg_wkly_wage']
    panel['wage_base'] = panel['fips'].map(wage_map_2015)
    panel['ln_wage_growth'] = np.log(panel['annual_avg_wkly_wage']) - np.log(panel['wage_base'])

    return panel


def print_descriptives(panel: pd.DataFrame, baseline: pd.DataFrame):
    """Print descriptive statistics table."""
    print("\n" + "=" * 70)
    print("TABLE: Descriptive Statistics — County-Level Sample")
    print("=" * 70)

    # Panel-level descriptives
    stats = []
    stats.append(('Annual avg. retail employment', panel['annual_avg_emplvl']))
    stats.append(('Annual avg. weekly wage ($)', panel['annual_avg_wkly_wage']))
    stats.append(('Annual avg. establishments', panel['annual_avg_estabs']))
    stats.append(('Log employment growth (vs 2015)', panel['ln_empl_growth']))
    stats.append(('Baseline wage 2015 ($)', baseline['annual_avg_wkly_wage']))
    stats.append(('Baseline employment 2015', baseline['annual_avg_emplvl']))

    print(f"  {'Variable':<38} {'N':>7} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 85)
    for label, series in stats:
        s = series.dropna()
        print(f"  {label:<38} {len(s):>7} {s.mean():>10.1f} {s.std():>10.1f} "
              f"{s.min():>10.1f} {s.max():>10.1f}")

    # E-commerce share over time
    print(f"\n  National e-commerce share (FRED ECOMPCTSA):")
    for yr in sorted(ECOM_SHARE.keys()):
        print(f"    {yr}: {ECOM_SHARE[yr]:.1f}%")

    # County counts by year
    print(f"\n  County-year observations by year:")
    for yr in sorted(panel['year'].unique()):
        n = len(panel[panel['year'] == yr])
        print(f"    {yr}: {n} counties")


def run_main_regression(panel: pd.DataFrame) -> dict:
    """Run the main specification: Eq. 14."""
    print("\n" + "=" * 70)
    print("MAIN SPECIFICATION: Good Jobs Buffer (Eq. 14)")
    print("  DV: ln(E_ct) - ln(E_c,2015)")
    print("  IV: W_z_c,2015 × Ecom_t")
    print("  FE: county + year | Clustered at county")
    print("  Note: Ecom_t main effect absorbed by year FE (national variable)")
    print("=" * 70)

    reg = panel.dropna(subset=['ln_empl_growth', 'wage_z_x_ecom']).copy()
    reg = reg.set_index(['fips', 'year'])

    mod = PanelOLS(reg['ln_empl_growth'], reg[['wage_z_x_ecom']],
                   entity_effects=True, time_effects=True, check_rank=False)
    r = mod.fit(cov_type='clustered', cluster_entity=True)

    n_counties = panel['fips'].nunique()
    b = r.params['wage_z_x_ecom']
    t_val = r.tstats['wage_z_x_ecom']
    p_val = r.pvalues['wage_z_x_ecom']

    print(f"\n  wage_z × ecom: β = {b:.4f}, t = {t_val:.2f}, p = {p_val:.4f}")
    print(f"  N = {r.nobs}, Counties = {n_counties}")
    print(f"  R² (within) = {r.rsquared_within:.4f}")

    # Interpretation
    print(f"\n  Interpretation: A county 1 SD above mean initial wage "
          f"(≈${panel['wage_2015'].std():.0f}/week)")
    print(f"  retains {b * 10 * 100:.1f} pp more employment per 10-point "
          f"increase in e-commerce share.")

    return {'spec': 'main', 'beta': b, 't': t_val, 'p': p_val,
            'n': int(r.nobs), 'n_counties': n_counties,
            'r2_within': r.rsquared_within}


def run_robustness(panel: pd.DataFrame) -> list:
    """Run robustness checks."""
    results = []

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECKS")
    print("=" * 70)

    def run_spec(data, label):
        reg = data.dropna(subset=['ln_empl_growth', 'wage_z_x_ecom']).copy()
        reg = reg.set_index(['fips', 'year'])
        mod = PanelOLS(reg['ln_empl_growth'], reg[['wage_z_x_ecom']],
                       entity_effects=True, time_effects=True, check_rank=False)
        r = mod.fit(cov_type='clustered', cluster_entity=True)
        b = r.params['wage_z_x_ecom']
        t_val = r.tstats['wage_z_x_ecom']
        p_val = r.pvalues['wage_z_x_ecom']
        n_c = data['fips'].nunique()
        print(f"\n  {label}:")
        print(f"    β = {b:.4f}, t = {t_val:.2f}, p = {p_val:.4f}, "
              f"N = {int(r.nobs)}, counties = {n_c}")
        results.append({'spec': label, 'beta': b, 't': t_val, 'p': p_val,
                        'n': int(r.nobs), 'n_counties': n_c})

    # 1. Exclude COVID years (2020-2021)
    no_covid = panel[~panel['year'].isin([2020, 2021])]
    run_spec(no_covid, "Excl. COVID (2020-2021)")

    # 2. Urban counties only (employment > median)
    median_empl = panel.groupby('fips')['annual_avg_emplvl'].mean().median()
    urban_fips = panel.groupby('fips')['annual_avg_emplvl'].mean()
    urban = panel[panel['fips'].isin(urban_fips[urban_fips > median_empl].index)]
    run_spec(urban, "Urban counties (empl > median)")

    # 3. Rural counties
    rural = panel[panel['fips'].isin(urban_fips[urban_fips <= median_empl].index)]
    run_spec(rural, "Rural counties (empl ≤ median)")

    # 4. Pre-period placebo (2015-2017 only)
    print("\n  Pre-period placebo (2015-2017):")
    pre = panel[panel['year'].isin([2015, 2016, 2017])]
    run_spec(pre, "Pre-period placebo (2015-2017)")

    # 5. Wage growth as DV (mechanism check)
    print("\n  Mechanism check: wage growth as DV")
    wage_reg = panel.dropna(subset=['ln_wage_growth', 'wage_z_x_ecom']).copy()
    wage_reg = wage_reg.set_index(['fips', 'year'])
    mod_w = PanelOLS(wage_reg['ln_wage_growth'], wage_reg[['wage_z_x_ecom']],
                     entity_effects=True, time_effects=True, check_rank=False)
    r_w = mod_w.fit(cov_type='clustered', cluster_entity=True)
    b_w = r_w.params['wage_z_x_ecom']
    t_w = r_w.tstats['wage_z_x_ecom']
    p_w = r_w.pvalues['wage_z_x_ecom']
    print(f"    β = {b_w:.4f}, t = {t_w:.2f}, p = {p_w:.4f}")
    print(f"    Effect on wage growth: {'null' if p_w > 0.10 else 'significant'}")
    results.append({'spec': 'wage_growth_DV', 'beta': b_w, 't': t_w, 'p': p_w,
                    'n': int(r_w.nobs), 'n_counties': panel['fips'].nunique()})

    # 6. Long-difference specification (2015 vs 2023)
    print("\n  Long-difference (2015 vs 2023):")
    d2015 = panel[panel['year'] == 2015][['fips', 'annual_avg_emplvl', 'wage_2015_z']].copy()
    d2023 = panel[panel['year'] == 2023][['fips', 'annual_avg_emplvl']].copy()
    d2023.columns = ['fips', 'empl_2023']
    ld = d2015.merge(d2023, on='fips', how='inner')
    ld['ln_growth'] = np.log(ld['empl_2023']) - np.log(ld['annual_avg_emplvl'])
    X_ld = sm.add_constant(ld['wage_2015_z'])
    mod_ld = sm.OLS(ld['ln_growth'], X_ld).fit(cov_type='HC1')
    b_ld = mod_ld.params['wage_2015_z']
    t_ld = mod_ld.tvalues['wage_2015_z']
    p_ld = mod_ld.pvalues['wage_2015_z']
    print(f"    β = {b_ld:.4f}, t = {t_ld:.2f}, p = {p_ld:.4f}, N = {len(ld)}")
    results.append({'spec': 'long_diff_2015_2023', 'beta': b_ld, 't': t_ld, 'p': p_ld,
                    'n': len(ld), 'n_counties': len(ld)})

    return results


def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # Load and filter
    panel, baseline = load_and_filter(project_root)

    # Build variables
    panel = build_regression_variables(panel, baseline)

    # Descriptive statistics
    print_descriptives(panel, baseline)

    # Main regression
    main_result = run_main_regression(panel)

    # Robustness
    rob_results = run_robustness(panel)

    # Save results
    all_results = [main_result] + rob_results
    results_df = pd.DataFrame(all_results)
    outpath = project_root / 'data' / 'processed' / 'county_regression_results.csv'
    results_df.to_csv(outpath, index=False)
    print(f"\n\nResults saved to {outpath}")

    # Print summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"  N = {main_result['n']:,} county-year observations")
    print(f"  Counties = {main_result['n_counties']:,}")
    print(f"  Main result: β₁ = {main_result['beta']:.4f} "
          f"(t = {main_result['t']:.2f}, p = {main_result['p']:.3f})")

    print("\n[DONE] County regressions complete.")


if __name__ == '__main__':
    main()
