#!/usr/bin/env python3
"""
14_cbp_establishment.py — Extensive-margin Good Jobs Buffer test.

Tests whether high-wage counties retain more retail establishments (not just
employment) as e-commerce grows. Establishment entry/exit is a visible
consequence of the Burnout Cliff — failed firms exit.

Uses QCEW annual_avg_estabs (already in the county panel) to test the
extensive margin alongside the intensive margin (employment).

Specification:
  ln Estab_{ct} - ln Estab_{c,2015} = α_c + γ_t + β₁(W^z_{c,2015} × Ecom_t) + ε

β₁ > 0 means the wage buffer operates on the extensive margin.
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import os
from pathlib import Path

ECOM_SHARE = {
    2015: 7.4, 2016: 8.1, 2017: 9.0, 2018: 9.9, 2019: 10.7,
    2020: 14.0, 2021: 13.2, 2022: 14.5, 2023: 15.4, 2024: 16.1,
}


def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    df = pd.read_csv('data/bls_qcew/retail_county_panel.csv')

    # Baseline filters (same as 11_county_regressions.py)
    baseline = df[df['year'] == 2015].copy()
    baseline_pass = baseline[
        (baseline['annual_avg_emplvl'] > 50) &
        (baseline['annual_avg_wkly_wage'] > 200)
    ]
    good_fips = baseline_pass['fips'].unique()
    panel = df[df['fips'].isin(good_fips)].copy()
    panel = panel[(panel['annual_avg_emplvl'] > 0) & (panel['annual_avg_wkly_wage'] > 0)]

    # Additional filter: need positive establishment counts
    panel = panel[panel['annual_avg_estabs'] > 0]
    print(f"Panel: {len(panel)} obs, {panel['fips'].nunique()} counties")

    # Build variables
    wage_2015 = baseline_pass.set_index('fips')['annual_avg_wkly_wage']
    w_mean, w_std = wage_2015.mean(), wage_2015.std()
    panel['wage_2015_z'] = (panel['fips'].map(wage_2015) - w_mean) / w_std

    estab_2015 = baseline_pass.set_index('fips')['annual_avg_estabs']
    empl_2015 = baseline_pass.set_index('fips')['annual_avg_emplvl']
    panel['estab_2015'] = panel['fips'].map(estab_2015)
    panel['empl_2015'] = panel['fips'].map(empl_2015)

    # Dependent variables
    panel['ln_estab_growth'] = np.log(panel['annual_avg_estabs']) - np.log(panel['estab_2015'])
    panel['ln_empl_growth'] = np.log(panel['annual_avg_emplvl']) - np.log(panel['empl_2015'])

    panel['ecom'] = panel['year'].map(ECOM_SHARE)
    panel['wage_z_x_ecom'] = panel['wage_2015_z'] * panel['ecom']

    # ── Descriptives ──
    print(f"\nBaseline (2015) establishments:")
    print(f"  Mean: {estab_2015.mean():.0f}, Median: {estab_2015.median():.0f}, "
          f"SD: {estab_2015.std():.0f}")
    print(f"  Range: {estab_2015.min():.0f} – {estab_2015.max():.0f}")

    # ── Specification 1: Employment buffer (replication) ──
    print("\n" + "=" * 70)
    print("SPECIFICATION 1: Employment buffer (replication of main result)")
    print("=" * 70)

    reg1 = panel.dropna(subset=['ln_empl_growth', 'wage_z_x_ecom']).copy()
    reg1 = reg1.set_index(['fips', 'year'])
    mod1 = PanelOLS(reg1['ln_empl_growth'], reg1[['wage_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r1 = mod1.fit(cov_type='clustered', cluster_entity=True)
    print(f"  Employment: β = {r1.params['wage_z_x_ecom']:.5f}, "
          f"t = {r1.tstats['wage_z_x_ecom']:.3f}, "
          f"p = {r1.pvalues['wage_z_x_ecom']:.4f}")

    # ── Specification 2: Establishment buffer ──
    print("\n" + "=" * 70)
    print("SPECIFICATION 2: Establishment buffer (extensive margin)")
    print("  DV: ln Estab_{ct} - ln Estab_{c,2015}")
    print("=" * 70)

    reg2 = panel.dropna(subset=['ln_estab_growth', 'wage_z_x_ecom']).copy()
    reg2 = reg2.set_index(['fips', 'year'])
    mod2 = PanelOLS(reg2['ln_estab_growth'], reg2[['wage_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r2 = mod2.fit(cov_type='clustered', cluster_entity=True)

    b2 = r2.params['wage_z_x_ecom']
    t2 = r2.tstats['wage_z_x_ecom']
    p2 = r2.pvalues['wage_z_x_ecom']
    print(f"  Establishments: β = {b2:.5f}, t = {t2:.3f}, p = {p2:.4f}")
    print(f"  R² (within) = {r2.rsquared_within:.4f}")

    # ── Specification 3: Exclude COVID ──
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Exclude COVID (2020-2021)")
    print("=" * 70)

    no_covid = panel[~panel['year'].isin([2020, 2021])].copy()
    reg3 = no_covid.dropna(subset=['ln_estab_growth', 'wage_z_x_ecom']).copy()
    reg3 = reg3.set_index(['fips', 'year'])
    mod3 = PanelOLS(reg3['ln_estab_growth'], reg3[['wage_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r3 = mod3.fit(cov_type='clustered', cluster_entity=True)
    print(f"  Estab (no COVID): β = {r3.params['wage_z_x_ecom']:.5f}, "
          f"t = {r3.tstats['wage_z_x_ecom']:.3f}, "
          f"p = {r3.pvalues['wage_z_x_ecom']:.4f}")

    # ── Specification 4: Urban/Rural split ──
    print("\n" + "=" * 70)
    print("URBAN/RURAL SPLIT")
    print("=" * 70)

    median_empl = panel.groupby('fips')['annual_avg_emplvl'].mean().median()
    urban_fips = panel.groupby('fips')['annual_avg_emplvl'].mean()

    for label, fips_set in [('Urban', urban_fips[urban_fips > median_empl].index),
                             ('Rural', urban_fips[urban_fips <= median_empl].index)]:
        sub = panel[panel['fips'].isin(fips_set)].copy()
        regs = sub.dropna(subset=['ln_estab_growth', 'wage_z_x_ecom']).copy()
        regs = regs.set_index(['fips', 'year'])
        mods = PanelOLS(regs['ln_estab_growth'], regs[['wage_z_x_ecom']],
                        entity_effects=True, time_effects=True, check_rank=False)
        rs = mods.fit(cov_type='clustered', cluster_entity=True)
        print(f"  {label}: β = {rs.params['wage_z_x_ecom']:.5f}, "
              f"t = {rs.tstats['wage_z_x_ecom']:.3f}, "
              f"p = {rs.pvalues['wage_z_x_ecom']:.4f}, "
              f"N = {rs.nobs:,}, counties = {sub['fips'].nunique():,}")

    # ── Specification 5: Employment per establishment (avg firm size) ──
    print("\n" + "=" * 70)
    print("MECHANISM: Employment per establishment (average firm size)")
    print("=" * 70)

    panel['emp_per_estab'] = panel['annual_avg_emplvl'] / panel['annual_avg_estabs']
    ep_2015 = panel[panel['year'] == 2015].set_index('fips')['emp_per_estab']
    panel['ln_ep_growth'] = np.log(panel['emp_per_estab']) - np.log(panel['fips'].map(ep_2015))

    reg5 = panel.dropna(subset=['ln_ep_growth', 'wage_z_x_ecom']).copy()
    reg5 = reg5.set_index(['fips', 'year'])
    mod5 = PanelOLS(reg5['ln_ep_growth'], reg5[['wage_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r5 = mod5.fit(cov_type='clustered', cluster_entity=True)
    print(f"  Emp/Estab: β = {r5.params['wage_z_x_ecom']:.5f}, "
          f"t = {r5.tstats['wage_z_x_ecom']:.3f}, "
          f"p = {r5.pvalues['wage_z_x_ecom']:.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: Intensive vs Extensive Margin")
    print("=" * 70)
    print(f"  Employment (intensive):   β = {r1.params['wage_z_x_ecom']:.5f}, "
          f"p = {r1.pvalues['wage_z_x_ecom']:.4f}")
    print(f"  Establishments (extensive): β = {b2:.5f}, p = {p2:.4f}")
    print(f"  Emp/Estab (firm size):    β = {r5.params['wage_z_x_ecom']:.5f}, "
          f"p = {r5.pvalues['wage_z_x_ecom']:.4f}")
    print(f"\n  N = {int(r2.nobs):,}, Counties = {panel['fips'].nunique():,}")

    print("\n[DONE] Establishment margin test complete.")


if __name__ == '__main__':
    main()
