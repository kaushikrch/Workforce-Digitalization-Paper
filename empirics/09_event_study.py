#!/usr/bin/env python3
"""
09_event_study.py — Technology-shock event study for WCT model.

Identification: Major omnichannel launches (BOPS/curbside) as firm-level
technology shocks. Tests whether firms with higher pre-shock workforce
investment (Glassdoor) experienced smaller ACSI declines post-shock.

Design: staggered DID with firm and year FE, pre-shock W as moderator.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t as t_dist
from linearmodels.panel import PanelOLS
import os, sys
from pathlib import Path

# ── Technology shock timeline (year of major omnichannel launch) ──
# Sources: annual reports, press releases, trade press
TECH_SHOCKS = {
    'WMT': 2017,   # Walmart Online Grocery Pickup national rollout
    'TGT': 2018,   # Target Drive Up + Shipt same-day integration
    'KR':  2016,   # Kroger ClickList (now Pickup) major expansion
    'HD':  2017,   # Home Depot BOPIS / curbside expansion
    'LOW': 2018,   # Lowe's omnichannel overhaul under Marvin Ellison
    'COST': 2017,  # Costco e-commerce replatform + Instacart partnership
    'BBY': 2016,   # Best Buy "Renew Blue" → BOPIS, ship-from-store
    'CVS': 2018,   # CVS digital health integration + Aetna merger
    'WBA': 2019,   # Walgreens digital pharmacy + drive-thru expansion
    'M':   2017,   # Macy's BOPIS + At Your Service expansion
    'KSS': 2017,   # Kohl's Amazon returns partnership + BOPIS
    'JWN': 2017,   # Nordstrom Local + BOPIS expansion
    'TJX': 2019,   # TJX e-commerce (late mover)
    'GPS': 2018,   # Gap BOPIS rollout across brands
    'DG':  2020,   # Dollar General DG Pickup pilot
    'DLTR': 2020,  # Dollar Tree online order rollout (late)
}

def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    df = pd.read_csv('data/processed/wct_empirical_panel.csv')
    print(f"Loaded panel: {len(df)} obs, {df['ticker'].nunique()} firms")

    # ── Construct event-study variables ──
    df['tech_shock_year'] = df['ticker'].map(TECH_SHOCKS)
    df['post_shock'] = (df['year'] >= df['tech_shock_year']).astype(int)
    df['event_time'] = df['year'] - df['tech_shock_year']

    # Pre-shock workforce level (average Glassdoor in [-3, -1] window before shock)
    def pre_shock_gd(group):
        shock_yr = group['tech_shock_year'].iloc[0]
        pre = group[(group['year'] >= shock_yr - 3) & (group['year'] < shock_yr)]
        return pre['glassdoor_overall'].mean()

    pre_gd = df.groupby('ticker').apply(pre_shock_gd, include_groups=False).rename('pre_gd')
    df = df.merge(pre_gd, on='ticker', how='left')

    # Interaction: post_shock × pre_workforce
    df['post_x_preW'] = df['post_shock'] * df['pre_gd']

    # Standardise for comparability
    for col in ['pre_gd', 'capex_intensity']:
        if col in df.columns:
            s = df[col].dropna()
            df[f'{col}_z'] = (df[col] - s.mean()) / s.std()

    df['post_x_preW_z'] = df['post_shock'] * df['pre_gd_z']

    # Pre-shock technology level
    def pre_shock_capex(group):
        shock_yr = group['tech_shock_year'].iloc[0]
        pre = group[(group['year'] >= shock_yr - 3) & (group['year'] < shock_yr)]
        return pre['capex_intensity'].mean()

    pre_kk = df.groupby('ticker').apply(pre_shock_capex, include_groups=False).rename('pre_K')
    df = df.merge(pre_kk, on='ticker', how='left')
    df['pre_K_z'] = (df['pre_K'] - df['pre_K'].mean()) / df['pre_K'].std()
    df['post_x_preK_z'] = df['post_shock'] * df['pre_K_z']

    # Triple interaction: post × preW × preK
    df['triple'] = df['post_shock'] * df['pre_gd_z'] * df['pre_K_z']

    # ── Filter to analysis window: [-3, +3] around shock ──
    window = df[(df['event_time'] >= -3) & (df['event_time'] <= 3)].copy()
    window = window.dropna(subset=['acsi_score', 'pre_gd_z'])
    print(f"\nEvent window [-3, +3]: {len(window)} obs, {window['ticker'].nunique()} firms")
    print(f"Pre-shock Glassdoor coverage: {window['pre_gd'].notna().sum()}/{len(window)}")

    n_firms = window['ticker'].nunique()
    df_eff = n_firms - 1

    # ── Specification 1: Basic DID — does ACSI change post-shock? ──
    print("\n" + "="*65)
    print("SPECIFICATION 1: Basic post-shock effect on ACSI")
    print("="*65)

    panel = window.set_index(['ticker', 'year'])
    mod1 = PanelOLS(panel['acsi_score'], panel[['post_shock']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r1 = mod1.fit(cov_type='clustered', cluster_entity=True)
    b = r1.params['post_shock']
    t_val = r1.tstats['post_shock']
    p_corr = 2 * (1 - t_dist.cdf(abs(t_val), df_eff))
    print(f"  post_shock: b={b:.3f}, t={t_val:.3f}, p[t({df_eff})]={p_corr:.4f}")

    # ── Specification 2: Post × pre-workforce interaction ──
    print("\n" + "="*65)
    print("SPECIFICATION 2: Post-shock × Pre-workforce Glassdoor")
    print("  Does higher pre-shock W buffer the ACSI impact?")
    print("="*65)

    mod2 = PanelOLS(panel['acsi_score'],
                    panel[['post_shock', 'post_x_preW_z']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r2 = mod2.fit(cov_type='clustered', cluster_entity=True)
    for var in ['post_shock', 'post_x_preW_z']:
        b = r2.params[var]
        t_val = r2.tstats[var]
        p_corr = 2 * (1 - t_dist.cdf(abs(t_val), df_eff))
        sig = '***' if p_corr < 0.01 else '**' if p_corr < 0.05 else '*' if p_corr < 0.10 else ''
        print(f"  {var}: b={b:.4f}, t={t_val:.3f}, p[t({df_eff})]={p_corr:.4f} {sig}")

    # ── Specification 3: Post × preW + Post × preK ──
    print("\n" + "="*65)
    print("SPECIFICATION 3: + Pre-shock technology control")
    print("="*65)

    mod3 = PanelOLS(panel['acsi_score'],
                    panel[['post_shock', 'post_x_preW_z', 'post_x_preK_z']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r3 = mod3.fit(cov_type='clustered', cluster_entity=True)
    for var in ['post_shock', 'post_x_preW_z', 'post_x_preK_z']:
        b = r3.params[var]
        t_val = r3.tstats[var]
        p_corr = 2 * (1 - t_dist.cdf(abs(t_val), df_eff))
        sig = '***' if p_corr < 0.01 else '**' if p_corr < 0.05 else '*' if p_corr < 0.10 else ''
        print(f"  {var}: b={b:.4f}, t={t_val:.3f}, p[t({df_eff})]={p_corr:.4f} {sig}")

    # ── Specification 4: Triple interaction (complementarity test) ──
    print("\n" + "="*65)
    print("SPECIFICATION 4: Triple interaction (post × preW × preK)")
    print("  Complementarity: does the W buffer matter MORE at high K?")
    print("="*65)

    w4 = window.dropna(subset=['acsi_score', 'pre_gd_z', 'pre_K_z']).copy()
    p4 = w4.set_index(['ticker', 'year'])
    if len(w4) >= 30:
        mod4 = PanelOLS(p4['acsi_score'],
                        p4[['post_shock', 'post_x_preW_z', 'post_x_preK_z', 'triple']],
                        entity_effects=True, time_effects=True, check_rank=False)
        r4 = mod4.fit(cov_type='clustered', cluster_entity=True)
        n4 = w4['ticker'].nunique()
        for var in ['post_shock', 'post_x_preW_z', 'post_x_preK_z', 'triple']:
            b = r4.params[var]
            t_val = r4.tstats[var]
            p_corr = 2 * (1 - t_dist.cdf(abs(t_val), n4 - 1))
            sig = '***' if p_corr < 0.01 else '**' if p_corr < 0.05 else '*' if p_corr < 0.10 else ''
            print(f"  {var}: b={b:.4f}, t={t_val:.3f}, p[t({n4-1})]={p_corr:.4f} {sig}")
    else:
        print(f"  Insufficient data ({len(w4)} obs)")

    # ── Event-time coefficients (parallel trends check) ──
    print("\n" + "="*65)
    print("PARALLEL TRENDS CHECK: Event-time dummies × pre-workforce")
    print("="*65)

    # Create event-time dummies (omit t=-1 as reference)
    for t in range(-3, 4):
        if t != -1:
            window[f'evt_{t}'] = (window['event_time'] == t).astype(int)
            window[f'evt_{t}_xW'] = window[f'evt_{t}'] * window['pre_gd_z']

    evt_cols = [f'evt_{t}' for t in range(-3, 4) if t != -1]
    evt_xW_cols = [f'evt_{t}_xW' for t in range(-3, 4) if t != -1]

    wpt = window.dropna(subset=['acsi_score'] + evt_cols + evt_xW_cols).copy()
    ppt = wpt.set_index(['ticker', 'year'])

    if len(wpt) >= 30:
        mod_pt = PanelOLS(ppt['acsi_score'],
                          ppt[evt_cols + evt_xW_cols],
                          entity_effects=True, check_rank=False)
        r_pt = mod_pt.fit(cov_type='clustered', cluster_entity=True)
        n_pt = wpt['ticker'].nunique()

        print(f"  {'Event time':<12} {'ACSI level':>12} {'× pre-W':>12} {'p(×W)':>10}")
        print("  " + "-"*50)
        for t in range(-3, 4):
            if t == -1:
                print(f"  {'t=-1 (ref)':<12} {'0.000':>12} {'0.000':>12} {'—':>10}")
                continue
            b_level = r_pt.params[f'evt_{t}']
            b_xW = r_pt.params[f'evt_{t}_xW']
            t_xW = r_pt.tstats[f'evt_{t}_xW']
            p_xW = 2 * (1 - t_dist.cdf(abs(t_xW), n_pt - 1))
            sig = '***' if p_xW < 0.01 else '**' if p_xW < 0.05 else '*' if p_xW < 0.10 else ''
            print(f"  t={t:<9} {b_level:>12.3f} {b_xW:>12.3f} {p_xW:>10.4f} {sig}")

        # Check pre-trends: joint F-test on pre-period interactions
        pre_xW = [f'evt_{t}_xW' for t in [-3, -2, 0]]  # pre-period (excl. ref)
        pre_coefs = [r_pt.params[v] for v in pre_xW[:2]]  # only t=-3, t=-2
        print(f"\n  Pre-trend check (t=-3, t=-2 × W coefficients):")
        for v in [f'evt_{-3}_xW', f'evt_{-2}_xW']:
            b = r_pt.params[v]
            t_v = r_pt.tstats[v]
            p_v = 2 * (1 - t_dist.cdf(abs(t_v), n_pt - 1))
            print(f"    {v}: b={b:.3f}, p={p_v:.3f}")

    print("\n[DONE] Event study complete.")

if __name__ == '__main__':
    main()
