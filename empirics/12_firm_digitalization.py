#!/usr/bin/env python3
"""
12_firm_digitalization.py — Firm-specific digitalization in the inverted-U
Uses digitalization_score_by_year.csv as firm-level F proxy.
"""
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# Load data
panel = pd.read_csv('data/processed/wct_empirical_panel.csv')
digi = pd.read_csv('data/store_level/digitalization_score_by_year.csv')

# Standardise brand names for merge
brand_map = {
    'Walmart': 'Walmart', 'Target': 'Target', 'Kroger': 'Kroger',
    'Costco': 'Costco', 'Home Depot': 'Home Depot', 'Lowes': "Lowe's",
    'CVS': 'CVS Health', 'Walgreens': 'Walgreens Boots Alliance',
    'Best Buy': 'Best Buy', 'Albertsons': 'Albertsons',
    'Dollar General': 'Dollar General', 'Dollar Tree': 'Dollar Tree',
    'Publix': 'Publix', 'Meijer': 'Meijer', 'Amazon': 'Amazon',
    'Whole Foods': 'Whole Foods Market', 'Nordstrom': 'Nordstrom'
}
digi['company'] = digi['brand'].map(brand_map).fillna(digi['brand'])

# Merge
merged = panel.merge(digi[['company', 'year', 'digitalization_pct']],
                     on=['company', 'year'], how='left')
merged = merged.dropna(subset=['acsi_score', 'digitalization_pct'])
merged['digi2'] = merged['digitalization_pct'] ** 2

print(f"\n{'='*60}")
print("FIRM-SPECIFIC DIGITALIZATION: INVERTED-U TEST")
print(f"{'='*60}")
print(f"Merged sample: N={len(merged)}, firms={merged['company'].nunique()}")
print(f"Digi range: {merged['digitalization_pct'].min():.1f}% — {merged['digitalization_pct'].max():.1f}%")
print(f"Digi mean: {merged['digitalization_pct'].mean():.1f}%, SD: {merged['digitalization_pct'].std():.1f}%")

# Set panel index
pdata = merged.set_index(['company', 'year'])

# Specification 1: Firm FE only (no year FE) — comparable to national Ecom spec
try:
    m1 = PanelOLS(pdata['acsi_score'],
                  pdata[['digitalization_pct', 'digi2']],
                  entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    G = merged['company'].nunique()
    from scipy.stats import t as tdist
    b2 = m1.params['digi2']
    se2 = m1.std_errors['digi2']
    t2 = b2 / se2
    p2 = 2 * (1 - tdist.cdf(abs(t2), G - 1))
    tp = -m1.params['digitalization_pct'] / (2 * b2) if b2 != 0 else np.nan
    print(f"\n--- Spec 1: Firm FE only ---")
    print(f"Digi:  b={m1.params['digitalization_pct']:.4f}, SE={m1.std_errors['digitalization_pct']:.4f}")
    print(f"Digi²: b={b2:.5f}, SE={se2:.5f}, t={t2:.3f}, p[t({G-1})]={p2:.4f}")
    print(f"Turning point: {tp:.1f}% digitalization")
    print(f"Within-R²: {m1.rsquared_within:.3f}")
    print(f"N={m1.nobs}, Firms={G}")
except Exception as e:
    print(f"Spec 1 failed: {e}")

# Specification 2: Firm + Year FE (now possible since Digi varies across firms)
try:
    m2 = PanelOLS(pdata['acsi_score'],
                  pdata[['digitalization_pct', 'digi2']],
                  entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
    b2 = m2.params['digi2']
    se2 = m2.std_errors['digi2']
    t2 = b2 / se2
    p2 = 2 * (1 - tdist.cdf(abs(t2), G - 1))
    tp = -m2.params['digitalization_pct'] / (2 * b2) if b2 != 0 else np.nan
    print(f"\n--- Spec 2: Firm + Year FE ---")
    print(f"Digi:  b={m2.params['digitalization_pct']:.4f}, SE={m2.std_errors['digitalization_pct']:.4f}")
    print(f"Digi²: b={b2:.5f}, SE={se2:.5f}, t={t2:.3f}, p[t({G-1})]={p2:.4f}")
    print(f"Turning point: {tp:.1f}% digitalization")
    print(f"Within-R²: {m2.rsquared_within:.3f}")
    print(f"N={m2.nobs}, Firms={G}")
except Exception as e:
    print(f"Spec 2 failed: {e}")

# Mediation analysis (Baron-Kenny)
print(f"\n{'='*60}")
print("MEDIATION ANALYSIS: Ecom → Glassdoor → ACSI")
print(f"{'='*60}")

med = panel.dropna(subset=['acsi_score', 'glassdoor_overall', 'ecom_share_pct']).copy()
med['ecom2'] = med['ecom_share_pct'] ** 2
print(f"Mediation sample: N={len(med)}, firms={med['company'].nunique()}")

mdata = med.set_index(['company', 'year'])

# Path c: Total effect (Ecom² → ACSI, no Glassdoor)
mc = PanelOLS(mdata['acsi_score'], mdata[['ecom_share_pct', 'ecom2']],
              entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

# Path a: Ecom → Glassdoor
ma = PanelOLS(mdata['glassdoor_overall'], mdata[['ecom_share_pct']],
              entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

# Path c': Direct effect (Ecom² → ACSI, controlling for Glassdoor)
mcp = PanelOLS(mdata['acsi_score'], mdata[['ecom_share_pct', 'ecom2', 'glassdoor_overall']],
               entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

print(f"\nPath c (total): Ecom² → ACSI = {mc.params['ecom2']:.5f} (p={mc.pvalues['ecom2']:.4f})")
print(f"Path a: Ecom → Glassdoor = {ma.params['ecom_share_pct']:.5f} (p={ma.pvalues['ecom_share_pct']:.4f})")
print(f"Path c' (direct): Ecom² → ACSI|GD = {mcp.params['ecom2']:.5f} (p={mcp.pvalues['ecom2']:.4f})")
print(f"Path b: Glassdoor → ACSI|Ecom = {mcp.params['glassdoor_overall']:.4f} (p={mcp.pvalues['glassdoor_overall']:.4f})")
attenuation = 1 - (mcp.params['ecom2'] / mc.params['ecom2']) if mc.params['ecom2'] != 0 else np.nan
print(f"\nAttenuation: {attenuation*100:.1f}% of total effect explained by workforce channel")

# JOLTS aggregate test
print(f"\n{'='*60}")
print("JOLTS: Retail Quits vs E-commerce Share")
print(f"{'='*60}")

macro = pd.read_csv('data/raw/macro_controls.csv')
macro = macro.dropna(subset=['retail_quits_rate', 'ecom_share_pct'])
print(f"JOLTS sample: T={len(macro)} years ({macro['year'].min()}-{macro['year'].max()})")

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

X = add_constant(macro['ecom_share_pct'])
m_jolts = OLS(macro['retail_quits_rate'], X).fit()
print(f"Quits = {m_jolts.params.iloc[0]:.3f} + {m_jolts.params.iloc[1]:.4f} × Ecom")
print(f"  Ecom coef: {m_jolts.params.iloc[1]:.4f}, SE={m_jolts.bse.iloc[1]:.4f}, t={m_jolts.tvalues.iloc[1]:.2f}, p={m_jolts.pvalues.iloc[1]:.4f}")
print(f"  R²={m_jolts.rsquared:.3f}")

# Also test turnover
m_turn = OLS(macro['retail_turnover_rate'], X).fit()
print(f"\nTurnover = {m_turn.params.iloc[0]:.3f} + {m_turn.params.iloc[1]:.4f} × Ecom")
print(f"  Ecom coef: {m_turn.params.iloc[1]:.4f}, SE={m_turn.bse.iloc[1]:.4f}, t={m_turn.tvalues.iloc[1]:.2f}, p={m_turn.pvalues.iloc[1]:.4f}")
print(f"  R²={m_turn.rsquared:.3f}")

print(f"\n{'='*60}")
print("[DONE] All three empirical tests complete.")
print(f"{'='*60}")
