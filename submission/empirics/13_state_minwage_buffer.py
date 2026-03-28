#!/usr/bin/env python3
"""
13_state_minwage_buffer.py — State minimum wage as policy instrument for
workforce investment (f) in the Good Jobs Buffer test.

Tests whether counties in states with higher minimum wages show more resilient
retail employment as e-commerce grows. The state minimum wage is a policy
instrument for f that is plausibly exogenous to individual firm decisions.

Specification:
  ln E_{ct} - ln E_{c,2015} = α_c + γ_t + β₁(MW^z_{s,t} × Ecom_t)
                                          + β₂(W^z_{c,2015} × Ecom_t) + ε_{ct}

MW_{s,t} is state minimum wage (standardised), varying by state and year.
β₁ > 0 means policy-driven workforce investment buffers e-commerce effects.

Data: BLS QCEW county panel (existing) + state minimum wage panel (constructed
from DOL/EPI public data).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t as t_dist
from linearmodels.panel import PanelOLS
import os
from pathlib import Path

# ── National e-commerce share (FRED ECOMPCTSA) ──
ECOM_SHARE = {
    2015: 7.4, 2016: 8.1, 2017: 9.0, 2018: 9.9, 2019: 10.7,
    2020: 14.0, 2021: 13.2, 2022: 14.5, 2023: 15.4, 2024: 16.1,
}

# ── State minimum wages 2015-2024 ($/hour, effective Jan 1 or as noted) ──
# Sources: DOL WHD, EPI Minimum Wage Tracker, state labor dept websites.
# Federal minimum = $7.25 throughout. States at or below federal use $7.25.
# State FIPS → {year: wage}. Only states above federal listed; rest default.
FEDERAL_MW = 7.25

# fmt: off
STATE_MW = {
    # Alaska (02)
    2: {2015: 8.75, 2016: 9.75, 2017: 9.80, 2018: 9.84, 2019: 9.89,
        2020: 10.19, 2021: 10.34, 2022: 10.34, 2023: 10.85, 2024: 11.73},
    # Arizona (04)
    4: {2015: 8.05, 2016: 8.05, 2017: 10.00, 2018: 10.50, 2019: 11.00,
        2020: 12.00, 2021: 12.15, 2022: 12.80, 2023: 13.85, 2024: 14.35},
    # Arkansas (05)
    5: {2015: 7.50, 2016: 8.00, 2017: 8.50, 2018: 8.50, 2019: 9.25,
        2020: 10.00, 2021: 11.00, 2022: 11.00, 2023: 11.00, 2024: 11.00},
    # California (06)
    6: {2015: 9.00, 2016: 10.00, 2017: 10.00, 2018: 10.50, 2019: 11.00,
        2020: 12.00, 2021: 13.00, 2022: 14.00, 2023: 15.00, 2024: 16.00},
    # Colorado (08)
    8: {2015: 8.23, 2016: 8.31, 2017: 9.30, 2018: 10.20, 2019: 11.10,
        2020: 12.00, 2021: 12.32, 2022: 12.56, 2023: 13.65, 2024: 14.42},
    # Connecticut (09)
    9: {2015: 9.15, 2016: 9.60, 2017: 10.10, 2018: 10.10, 2019: 10.10,
        2020: 12.00, 2021: 13.00, 2022: 14.00, 2023: 15.00, 2024: 15.69},
    # Delaware (10)
    10: {2015: 8.25, 2016: 8.25, 2017: 8.25, 2018: 8.25, 2019: 8.75,
         2020: 9.25, 2021: 9.25, 2022: 10.50, 2023: 11.75, 2024: 13.25},
    # DC (11)
    11: {2015: 10.50, 2016: 11.50, 2017: 12.50, 2018: 13.25, 2019: 14.00,
         2020: 15.00, 2021: 15.20, 2022: 15.20, 2023: 17.00, 2024: 17.50},
    # Florida (12)
    12: {2015: 8.05, 2016: 8.05, 2017: 8.10, 2018: 8.25, 2019: 8.46,
         2020: 8.56, 2021: 10.00, 2022: 11.00, 2023: 12.00, 2024: 13.00},
    # Hawaii (15)
    15: {2015: 7.75, 2016: 8.50, 2017: 9.25, 2018: 10.10, 2019: 10.10,
         2020: 10.10, 2021: 10.10, 2022: 12.00, 2023: 14.00, 2024: 14.00},
    # Illinois (17)
    17: {2015: 8.25, 2016: 8.25, 2017: 8.25, 2018: 8.25, 2019: 8.25,
         2020: 10.00, 2021: 11.00, 2022: 12.00, 2023: 13.00, 2024: 14.00},
    # Maine (23)
    23: {2015: 7.50, 2016: 7.50, 2017: 9.00, 2018: 10.00, 2019: 11.00,
         2020: 12.00, 2021: 12.15, 2022: 12.75, 2023: 13.80, 2024: 14.15},
    # Maryland (24)
    24: {2015: 8.00, 2016: 8.75, 2017: 9.25, 2018: 10.10, 2019: 10.10,
         2020: 11.00, 2021: 11.75, 2022: 12.50, 2023: 13.25, 2024: 15.00},
    # Massachusetts (25)
    25: {2015: 9.00, 2016: 10.00, 2017: 11.00, 2018: 11.00, 2019: 12.00,
         2020: 12.75, 2021: 13.50, 2022: 14.25, 2023: 15.00, 2024: 15.00},
    # Michigan (26)
    26: {2015: 8.15, 2016: 8.50, 2017: 8.90, 2018: 9.25, 2019: 9.45,
         2020: 9.65, 2021: 9.87, 2022: 9.87, 2023: 10.10, 2024: 10.33},
    # Minnesota (27)
    27: {2015: 8.00, 2016: 9.00, 2017: 9.50, 2018: 9.65, 2019: 9.86,
         2020: 10.00, 2021: 10.08, 2022: 10.33, 2023: 10.59, 2024: 10.85},
    # Missouri (29)
    29: {2015: 7.65, 2016: 7.65, 2017: 7.70, 2018: 7.85, 2019: 8.60,
         2020: 9.45, 2021: 10.30, 2022: 11.15, 2023: 12.00, 2024: 12.30},
    # Montana (30)
    30: {2015: 8.05, 2016: 8.05, 2017: 8.15, 2018: 8.30, 2019: 8.50,
         2020: 8.65, 2021: 8.75, 2022: 9.20, 2023: 9.95, 2024: 10.30},
    # Nebraska (31)
    31: {2015: 8.00, 2016: 9.00, 2017: 9.00, 2018: 9.00, 2019: 9.00,
         2020: 9.00, 2021: 9.00, 2022: 9.00, 2023: 10.50, 2024: 12.00},
    # Nevada (32)
    32: {2015: 8.25, 2016: 8.25, 2017: 8.25, 2018: 8.25, 2019: 8.25,
         2020: 8.25, 2021: 8.75, 2022: 9.50, 2023: 10.50, 2024: 12.00},
    # New Jersey (34)
    34: {2015: 8.38, 2016: 8.38, 2017: 8.44, 2018: 8.60, 2019: 8.85,
         2020: 11.00, 2021: 12.00, 2022: 13.00, 2023: 14.13, 2024: 15.13},
    # New Mexico (35)
    35: {2015: 7.50, 2016: 7.50, 2017: 7.50, 2018: 7.50, 2019: 7.50,
         2020: 9.00, 2021: 10.50, 2022: 11.50, 2023: 12.00, 2024: 12.00},
    # New York (36)
    36: {2015: 8.75, 2016: 9.00, 2017: 9.70, 2018: 10.40, 2019: 11.10,
         2020: 11.80, 2021: 12.50, 2022: 13.20, 2023: 14.20, 2024: 15.00},
    # Ohio (39)
    39: {2015: 8.10, 2016: 8.10, 2017: 8.15, 2018: 8.30, 2019: 8.55,
         2020: 8.70, 2021: 8.80, 2022: 9.30, 2023: 10.10, 2024: 10.45},
    # Oregon (41)
    41: {2015: 9.25, 2016: 9.25, 2017: 9.75, 2018: 10.25, 2019: 10.75,
         2020: 11.25, 2021: 12.00, 2022: 12.50, 2023: 13.50, 2024: 14.70},
    # Rhode Island (44)
    44: {2015: 9.00, 2016: 9.60, 2017: 9.60, 2018: 10.10, 2019: 10.50,
         2020: 10.50, 2021: 11.50, 2022: 12.25, 2023: 13.00, 2024: 14.00},
    # South Dakota (46)
    46: {2015: 8.50, 2016: 8.55, 2017: 8.65, 2018: 8.85, 2019: 9.10,
         2020: 9.30, 2021: 9.45, 2022: 9.95, 2023: 10.80, 2024: 11.20},
    # Vermont (50)
    50: {2015: 9.15, 2016: 9.60, 2017: 10.00, 2018: 10.50, 2019: 10.78,
         2020: 10.96, 2021: 11.75, 2022: 12.55, 2023: 13.18, 2024: 13.67},
    # Virginia (51)
    51: {2015: 7.25, 2016: 7.25, 2017: 7.25, 2018: 7.25, 2019: 7.25,
         2020: 7.25, 2021: 9.50, 2022: 11.00, 2023: 12.00, 2024: 12.00},
    # Washington (53)
    53: {2015: 9.47, 2016: 9.47, 2017: 11.00, 2018: 11.50, 2019: 12.00,
         2020: 13.50, 2021: 13.69, 2022: 14.49, 2023: 15.74, 2024: 16.28},
    # West Virginia (54) — at federal
    # No entry needed, defaults to FEDERAL_MW
}
# fmt: on


def get_state_mw(state_fips: int, year: int) -> float:
    """Return state minimum wage; federal floor if no state data."""
    if state_fips in STATE_MW:
        return STATE_MW[state_fips].get(year, FEDERAL_MW)
    return FEDERAL_MW


def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # ── Load QCEW county panel (reuse 11_county_regressions.py logic) ──
    df = pd.read_csv('data/bls_qcew/retail_county_panel.csv')
    print(f"Raw QCEW: {len(df)} rows, {df['fips'].nunique()} counties")

    # Baseline filters
    baseline = df[df['year'] == 2015].copy()
    baseline_pass = baseline[
        (baseline['annual_avg_emplvl'] > 50) &
        (baseline['annual_avg_wkly_wage'] > 200)
    ]
    good_fips = baseline_pass['fips'].unique()
    panel = df[df['fips'].isin(good_fips)].copy()
    panel = panel[(panel['annual_avg_emplvl'] > 0) & (panel['annual_avg_wkly_wage'] > 0)]
    print(f"Filtered: {len(panel)} obs, {panel['fips'].nunique()} counties")

    # ── Extract state FIPS, merge minimum wage ──
    panel['state_fips'] = panel['fips'] // 1000
    panel['mw'] = panel.apply(lambda r: get_state_mw(r['state_fips'], r['year']), axis=1)

    # Standardise MW (across all state-years)
    mw_mean = panel['mw'].mean()
    mw_std = panel['mw'].std()
    panel['mw_z'] = (panel['mw'] - mw_mean) / mw_std
    print(f"\nMinimum wage: mean=${mw_mean:.2f}/hr, SD=${mw_std:.2f}/hr")
    print(f"  Range: ${panel['mw'].min():.2f} – ${panel['mw'].max():.2f}")
    print(f"  States above federal: {(panel.groupby('state_fips')['mw'].mean() > FEDERAL_MW).sum()}")

    # ── Build regression variables ──
    wage_2015 = baseline_pass.set_index('fips')['annual_avg_wkly_wage']
    w_mean, w_std = wage_2015.mean(), wage_2015.std()
    panel['wage_2015'] = panel['fips'].map(wage_2015)
    panel['wage_2015_z'] = (panel['wage_2015'] - w_mean) / w_std

    empl_2015 = baseline_pass.set_index('fips')['annual_avg_emplvl']
    panel['empl_2015'] = panel['fips'].map(empl_2015)
    panel['ln_empl_growth'] = np.log(panel['annual_avg_emplvl']) - np.log(panel['empl_2015'])

    panel['ecom'] = panel['year'].map(ECOM_SHARE)
    panel['wage_z_x_ecom'] = panel['wage_2015_z'] * panel['ecom']
    panel['mw_z_x_ecom'] = panel['mw_z'] * panel['ecom']

    # MW change since 2015 (alternative: Δ specification)
    mw_2015 = panel[panel['year'] == 2015].set_index('fips')['mw']
    panel['mw_2015'] = panel['fips'].map(mw_2015)
    panel['mw_change'] = panel['mw'] - panel['mw_2015']
    panel['mw_change_z'] = (panel['mw_change'] - panel['mw_change'].mean()) / panel['mw_change'].std()
    panel['mw_change_z_x_ecom'] = panel['mw_change_z'] * panel['ecom']

    # ── Specification 1: MW level × Ecom (main) ──
    print("\n" + "=" * 70)
    print("SPECIFICATION 1: State Minimum Wage × E-commerce (level)")
    print("  DV: ln E_ct - ln E_{c,2015}")
    print("  IV: MW^z_{s,t} × Ecom_t")
    print("  FE: county + year | Clustered at state")
    print("=" * 70)

    reg = panel.dropna(subset=['ln_empl_growth', 'mw_z_x_ecom']).copy()
    reg = reg.set_index(['fips', 'year'])

    mod1 = PanelOLS(reg['ln_empl_growth'], reg[['mw_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    # Cluster at state level (more conservative than county)
    # linearmodels requires clusters as a Series with matching MultiIndex
    state_cluster = reg.index.get_level_values('fips').map(lambda x: x // 1000)
    state_cluster = pd.Series(state_cluster.values, index=reg.index, name='state')
    r1 = mod1.fit(cov_type='clustered', clusters=state_cluster)

    b1 = r1.params['mw_z_x_ecom']
    t1 = r1.tstats['mw_z_x_ecom']
    p1 = r1.pvalues['mw_z_x_ecom']
    n_states = panel['state_fips'].nunique()
    print(f"\n  MW^z × Ecom: β = {b1:.5f}, t = {t1:.3f}, p = {p1:.4f}")
    print(f"  N = {r1.nobs:,}, Counties = {panel['fips'].nunique():,}, States = {n_states}")
    print(f"  R² (within) = {r1.rsquared_within:.4f}")

    # ── Specification 2: MW + Wage (horse race) ──
    print("\n" + "=" * 70)
    print("SPECIFICATION 2: MW × Ecom + Wage × Ecom (horse race)")
    print("  Tests whether MW effect survives controlling for initial county wages")
    print("=" * 70)

    reg2 = panel.dropna(subset=['ln_empl_growth', 'mw_z_x_ecom', 'wage_z_x_ecom']).copy()
    reg2_idx = reg2.set_index(['fips', 'year'])

    mod2 = PanelOLS(reg2_idx['ln_empl_growth'],
                    reg2_idx[['mw_z_x_ecom', 'wage_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    sc2 = pd.Series(reg2_idx.index.get_level_values('fips').map(lambda x: x // 1000).values,
                     index=reg2_idx.index, name='state')
    r2 = mod2.fit(cov_type='clustered', clusters=sc2)

    for var in ['mw_z_x_ecom', 'wage_z_x_ecom']:
        b = r2.params[var]
        t = r2.tstats[var]
        p = r2.pvalues[var]
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f"  {var}: β = {b:.5f}, t = {t:.3f}, p = {p:.4f} {sig}")
    print(f"  R² (within) = {r2.rsquared_within:.4f}")

    # ── Specification 3: MW change × Ecom ──
    print("\n" + "=" * 70)
    print("SPECIFICATION 3: MW change since 2015 × Ecom")
    print("  ΔMW identifies states that raised MW during the period")
    print("=" * 70)

    reg3 = panel.dropna(subset=['ln_empl_growth', 'mw_change_z_x_ecom']).copy()
    reg3_idx = reg3.set_index(['fips', 'year'])

    mod3 = PanelOLS(reg3_idx['ln_empl_growth'], reg3_idx[['mw_change_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    sc3 = pd.Series(reg3_idx.index.get_level_values('fips').map(lambda x: x // 1000).values,
                     index=reg3_idx.index, name='state')
    r3 = mod3.fit(cov_type='clustered', clusters=sc3)

    b3 = r3.params['mw_change_z_x_ecom']
    t3 = r3.tstats['mw_change_z_x_ecom']
    p3 = r3.pvalues['mw_change_z_x_ecom']
    print(f"\n  ΔMW^z × Ecom: β = {b3:.5f}, t = {t3:.3f}, p = {p3:.4f}")

    # ── Robustness: Exclude COVID ──
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Exclude COVID (2020-2021)")
    print("=" * 70)

    no_covid = panel[~panel['year'].isin([2020, 2021])].copy()
    reg4 = no_covid.dropna(subset=['ln_empl_growth', 'mw_z_x_ecom']).copy()
    reg4_idx = reg4.set_index(['fips', 'year'])

    mod4 = PanelOLS(reg4_idx['ln_empl_growth'], reg4_idx[['mw_z_x_ecom']],
                    entity_effects=True, time_effects=True, check_rank=False)
    sc4 = pd.Series(reg4_idx.index.get_level_values('fips').map(lambda x: x // 1000).values,
                     index=reg4_idx.index, name='state')
    r4 = mod4.fit(cov_type='clustered', clusters=sc4)

    b4 = r4.params['mw_z_x_ecom']
    t4 = r4.tstats['mw_z_x_ecom']
    p4 = r4.pvalues['mw_z_x_ecom']
    print(f"  MW^z × Ecom: β = {b4:.5f}, t = {t4:.3f}, p = {p4:.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Spec 1 (MW level):   β = {b1:.5f}, p = {p1:.4f}")
    print(f"  Spec 2 (horse race): MW β = {r2.params['mw_z_x_ecom']:.5f}, "
          f"Wage β = {r2.params['wage_z_x_ecom']:.5f}")
    print(f"  Spec 3 (ΔMW):       β = {b3:.5f}, p = {p3:.4f}")
    print(f"  Robustness (no COVID): β = {b4:.5f}, p = {p4:.4f}")
    print(f"\n  Clustered at state level ({n_states} clusters)")
    print(f"  N = {int(r1.nobs):,} county-year obs, {panel['fips'].nunique():,} counties")

    print("\n[DONE] State minimum wage buffer test complete.")


if __name__ == '__main__':
    main()
