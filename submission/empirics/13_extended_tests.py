#!/usr/bin/env python3
"""
13_extended_tests.py — Four extended empirical tests for WCT model.

Test 1: Cross-lagged panel — Glassdoor_{t-1} → ACSI_t (W leads T prediction)
Test 2: Post-launch ACSI persistence by ΔF size (Burnout Cliff proxy)
Test 3: Pre-launch workforce/CapEx ratio → post-launch resilience (Corollary 5 proxy)
Test 4: Technology count quadratic → ACSI (b > 1 proxy, firm-level F variation)

All tests use t(G-1) degrees of freedom for clustered inference.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from linearmodels.panel import PanelOLS
from scipy.stats import t as t_dist
import os
from pathlib import Path

# ── Setup ──
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)

panel = pd.read_csv('data/processed/wct_empirical_panel.csv')
panel_v2 = pd.read_csv('data/processed/wct_empirical_panel_v2.csv')
digi = pd.read_csv('data/store_level/digitalization_score_by_year.csv')
tech_timeline = pd.read_csv('data/store_level/technology_adoption_timeline.csv')

# ── Technology shock timeline (from 09_event_study.py) ──
TECH_SHOCKS = {
    'WMT': 2017, 'TGT': 2018, 'KR': 2016, 'HD': 2017, 'LOW': 2018,
    'COST': 2017, 'BBY': 2016, 'CVS': 2018, 'WBA': 2019, 'M': 2017,
    'KSS': 2017, 'JWN': 2017, 'TJX': 2019, 'GPS': 2018, 'DG': 2020,
    'DLTR': 2020,
}

brand_map = {
    'Walmart': 'Walmart', 'Target': 'Target', 'Kroger': 'Kroger',
    'Costco': 'Costco', 'Home Depot': 'Home Depot', 'Lowes': "Lowe's",
}

# Brand-to-ticker for technology timeline merges
brand_ticker = {
    'Walmart': 'WMT', 'Target': 'TGT', 'Kroger': 'KR',
    'Costco': 'COST', 'Home Depot': 'HD', "Lowe's": 'LOW',
}


def pval_corrected(t_val, n_clusters):
    """Two-sided p-value with t(G-1) degrees of freedom."""
    return 2 * (1 - t_dist.cdf(abs(t_val), n_clusters - 1))


def sig_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def report_coef(name, result, var, n_clusters):
    """Report a single coefficient with t(G-1) correction."""
    b = result.params[var]
    t_val = result.tstats[var]
    p = pval_corrected(t_val, n_clusters)
    print(f"  {name}: b={b:.4f}, t={t_val:.3f}, p[t({n_clusters-1})]={p:.4f} {sig_stars(p)}")
    return b, t_val, p


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Cross-Lagged Panel — W Leads T
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST 1: CROSS-LAGGED PANEL — DOES W (GLASSDOOR) LEAD T (ACSI)?")
print("=" * 70)
print("Theory prediction: workforce well-being deteriorates before trust.")
print("Test: Glassdoor_{t-1} → ACSI_t after controlling for ACSI_{t-1}.\n")

# Sort and create lags
panel_sorted = panel.sort_values(['company', 'year'])
panel_sorted['glassdoor_lag1'] = panel_sorted.groupby('company')['glassdoor_overall'].shift(1)
panel_sorted['acsi_lag1'] = panel_sorted.groupby('company')['acsi_score'].shift(1)

# Forward direction: ACSI_t ~ Glassdoor_{t-1} + ACSI_{t-1}
cl_fwd = panel_sorted.dropna(subset=['acsi_score', 'glassdoor_lag1', 'acsi_lag1']).copy()
n_firms_fwd = cl_fwd['company'].nunique()
n_obs_fwd = len(cl_fwd)
print(f"Forward: ACSI_t ~ GD_{{t-1}} + ACSI_{{t-1}}")
print(f"  N={n_obs_fwd}, firms={n_firms_fwd}")

pdata_fwd = cl_fwd.set_index(['company', 'year'])
mod_fwd = PanelOLS(pdata_fwd['acsi_score'],
                   pdata_fwd[['glassdoor_lag1', 'acsi_lag1']],
                   entity_effects=True, time_effects=True, check_rank=False)
r_fwd = mod_fwd.fit(cov_type='clustered', cluster_entity=True)
b_gd_fwd, t_gd_fwd, p_gd_fwd = report_coef("Glassdoor_{t-1}", r_fwd, 'glassdoor_lag1', n_firms_fwd)
b_acsi_fwd, t_acsi_fwd, p_acsi_fwd = report_coef("ACSI_{t-1}", r_fwd, 'acsi_lag1', n_firms_fwd)

# Forward without year FE (robustness)
print(f"\n  Robustness: without year FE")
mod_fwd_noyr = PanelOLS(pdata_fwd['acsi_score'],
                        pdata_fwd[['glassdoor_lag1', 'acsi_lag1']],
                        entity_effects=True, time_effects=False, check_rank=False)
r_fwd_noyr = mod_fwd_noyr.fit(cov_type='clustered', cluster_entity=True)
report_coef("Glassdoor_{t-1} (no year FE)", r_fwd_noyr, 'glassdoor_lag1', n_firms_fwd)

# Reverse direction: Glassdoor_t ~ ACSI_{t-1} + Glassdoor_{t-1}
cl_rev = panel_sorted.dropna(subset=['glassdoor_overall', 'acsi_lag1', 'glassdoor_lag1']).copy()
# Need glassdoor_lag1 for the reverse too
cl_rev = cl_rev.dropna(subset=['glassdoor_overall', 'acsi_lag1', 'glassdoor_lag1'])
n_firms_rev = cl_rev['company'].nunique()
n_obs_rev = len(cl_rev)
print(f"\nReverse: GD_t ~ ACSI_{{t-1}} + GD_{{t-1}}")
print(f"  N={n_obs_rev}, firms={n_firms_rev}")

pdata_rev = cl_rev.set_index(['company', 'year'])
mod_rev = PanelOLS(pdata_rev['glassdoor_overall'],
                   pdata_rev[['acsi_lag1', 'glassdoor_lag1']],
                   entity_effects=True, time_effects=True, check_rank=False)
r_rev = mod_rev.fit(cov_type='clustered', cluster_entity=True)
b_acsi_rev, t_acsi_rev, p_acsi_rev = report_coef("ACSI_{t-1}", r_rev, 'acsi_lag1', n_firms_rev)
b_gd_rev, t_gd_rev, p_gd_rev = report_coef("Glassdoor_{t-1}", r_rev, 'glassdoor_lag1', n_firms_rev)

print(f"\n  SUMMARY: W→T (β={b_gd_fwd:.3f}, p={p_gd_fwd:.3f}) vs T→W (γ={b_acsi_rev:.4f}, p={p_acsi_rev:.3f})")
if p_gd_fwd < 0.10 and p_acsi_rev > 0.10:
    print("  → Glassdoor predicts ACSI but not vice versa: consistent with W leads T.")
elif p_gd_fwd > 0.10 and p_acsi_rev < 0.10:
    print("  → ACSI predicts Glassdoor but not vice versa: reverse direction.")
elif p_gd_fwd < 0.10 and p_acsi_rev < 0.10:
    print("  → Both directions significant: bidirectional Granger-causality (or common cause).")
else:
    print("  → Neither direction significant: insufficient power at annual frequency.")
print("  CAVEAT: Nickell bias from lagged DV with firm FE (O(1/T) ≈ 6-10%); annual data.")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Post-Launch ACSI Persistence — Burnout Cliff Proxy
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("TEST 2: POST-LAUNCH ACSI PERSISTENCE — BURNOUT CLIFF PROXY")
print("=" * 70)
print("Theory prediction: large-ΔF firms show deepening ACSI decline (t+3 < t+1).")
print("Bistability → persistent collapse, not transient adjustment.\n")

# Construct event-study variables
df = panel.copy()
df['tech_shock_year'] = df['ticker'].map(TECH_SHOCKS)
df['event_time'] = df['year'] - df['tech_shock_year']

# Event window [-3, +3], exclude Amazon (no discrete launch)
window = df[(df['event_time'] >= -3) & (df['event_time'] <= 3)].copy()
window = window.dropna(subset=['acsi_score', 'tech_shock_year'])
n_firms_es = window['ticker'].nunique()
print(f"Event window [-3,+3]: N={len(window)}, firms={n_firms_es}")

# --- Approach A: Slope test of post-shock trajectory ---
print("\n--- Approach A: Is the post-shock trajectory monotonically declining? ---")

# Create event-time dummies (omit t=-1)
for t in range(-3, 4):
    if t != -1:
        window[f'evt_{t}'] = (window['event_time'] == t).astype(int)

evt_cols = [f'evt_{t}' for t in range(-3, 4) if t != -1]
wpt = window.dropna(subset=['acsi_score'] + evt_cols).copy()
ppt = wpt.set_index(['ticker', 'year'])

mod_evt = PanelOLS(ppt['acsi_score'], ppt[evt_cols],
                   entity_effects=True, check_rank=False)
r_evt = mod_evt.fit(cov_type='clustered', cluster_entity=True)
n_cl = wpt['ticker'].nunique()

print(f"  Event-time coefficients (reference: t=-1):")
for t in range(-3, 4):
    if t == -1:
        print(f"    t={t:+d}: (reference)")
        continue
    b = r_evt.params[f'evt_{t}']
    tv = r_evt.tstats[f'evt_{t}']
    p = pval_corrected(tv, n_cl)
    print(f"    t={t:+d}: δ={b:.3f}, t={tv:.3f}, p={p:.3f} {sig_stars(p)}")

# Slope test: is δ_{+3} - δ_{+1} < 0?
d1 = r_evt.params['evt_1']
d3 = r_evt.params['evt_3']
diff = d3 - d1
print(f"\n  Slope test: δ_{{+3}} - δ_{{+1}} = {d3:.3f} - {d1:.3f} = {diff:.3f}")
print(f"  (negative = deepening decline, consistent with slow collapse)")
# Joint test: post-trend linear combination
# Use approximate Wald: diff / sqrt(var(d3) + var(d1) - 2*cov(d3,d1))
vcov = r_evt.cov
se_diff = np.sqrt(vcov.loc['evt_3', 'evt_3'] + vcov.loc['evt_1', 'evt_1']
                  - 2 * vcov.loc['evt_3', 'evt_1'])
t_diff = diff / se_diff
p_diff = pval_corrected(t_diff, n_cl)
print(f"  t-stat(diff) = {t_diff:.3f}, p[t({n_cl-1})] = {p_diff:.3f} {sig_stars(p_diff)}")
if diff < 0:
    print("  → Post-shock trajectory deepening: consistent with Burnout Cliff (slow collapse).")
else:
    print("  → Post-shock trajectory flat or recovering: inconsistent with persistent collapse.")

# --- Approach B: High-ΔF vs Low-ΔF firms ---
print("\n--- Approach B: High-ΔF vs. Low-ΔF firms (technology count around launch) ---")

# Count technologies launched within ±1 year of shock
tech_counts = {}
for _, row in tech_timeline.iterrows():
    brand = row['brand']
    ticker = brand_ticker.get(brand)
    if ticker is None:
        continue
    shock_yr = TECH_SHOCKS.get(ticker)
    if shock_yr is None:
        continue
    # Technologies launched within ±1 year of the shock
    if abs(row['year'] - shock_yr) <= 1:
        tech_counts[ticker] = tech_counts.get(ticker, 0) + 1

# For firms not in technology_adoption_timeline, use default of 1
for ticker in TECH_SHOCKS:
    if ticker not in tech_counts:
        tech_counts[ticker] = 1  # single launch assumed

tc_df = pd.DataFrame({'ticker': list(tech_counts.keys()),
                       'delta_f_count': list(tech_counts.values())})
print(f"  Technology count (within ±1yr of shock):")
for _, r in tc_df.sort_values('delta_f_count', ascending=False).iterrows():
    print(f"    {r['ticker']}: {r['delta_f_count']} technologies")

median_df = tc_df['delta_f_count'].median()
tc_df['high_df'] = (tc_df['delta_f_count'] > median_df).astype(int)
print(f"  Median ΔF count: {median_df}")
print(f"  High-ΔF firms (>{median_df}): {tc_df['high_df'].sum()}, Low: {(1-tc_df['high_df']).sum()}")

# Merge into event window
window2 = window.merge(tc_df[['ticker', 'high_df']], on='ticker', how='left')
window2['high_df'] = window2['high_df'].fillna(0).astype(int)
window2['post'] = (window2['event_time'] >= 0).astype(int)
window2['post_late'] = (window2['event_time'] >= 2).astype(int)
window2['post_x_highdf'] = window2['post'] * window2['high_df']
window2['postlate_x_highdf'] = window2['post_late'] * window2['high_df']

w2 = window2.dropna(subset=['acsi_score']).copy()
p2 = w2.set_index(['ticker', 'year'])
n_cl2 = w2['ticker'].nunique()

mod_df = PanelOLS(p2['acsi_score'],
                  p2[['post', 'post_late', 'post_x_highdf', 'postlate_x_highdf']],
                  entity_effects=True, check_rank=False)
r_df = mod_df.fit(cov_type='clustered', cluster_entity=True)

print(f"\n  Specification: ACSI ~ Post + PostLate + Post×HighΔF + PostLate×HighΔF + FE")
print(f"  N={len(w2)}, firms={n_cl2}")
for var in ['post', 'post_late', 'post_x_highdf', 'postlate_x_highdf']:
    report_coef(var, r_df, var, n_cl2)

print("  CAVEAT: N=16 firms, very low-powered; ΔF proxy is coarse (tech count).")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Pre-Launch Workforce/CapEx → Post-Launch Resilience
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("TEST 3: PRE-LAUNCH WORKFORCE/CAPEX → POST-LAUNCH RESILIENCE")
print("=" * 70)
print("Theory prediction (Corollary 5): higher pre-launch workforce investment")
print("relative to technology investment → less ACSI decline post-launch.\n")

# Build event-study panel with workforce proxies
df3 = panel_v2.copy()
df3['tech_shock_year'] = df3['ticker'].map(TECH_SHOCKS)
df3['event_time'] = df3['year'] - df3['tech_shock_year']

# Compute SGA/CapEx ratio
df3['sga_capex_ratio'] = df3['sga_intensity'] / df3['capex_intensity']
# Replace inf/nan from division
df3['sga_capex_ratio'] = df3['sga_capex_ratio'].replace([np.inf, -np.inf], np.nan)

# Pre-shock means (window [-3, -1])
proxy_cols = ['sga_capex_ratio', 'rev_per_employee']
# Also try emp_liab_current_rev from v2 if available
if 'emp_liab_current_rev' in df3.columns:
    proxy_cols.append('emp_liab_current_rev')

pre_means = {}
for proxy in proxy_cols:
    def _pre_mean(group, proxy=proxy):
        shock_yr = group['tech_shock_year'].iloc[0]
        pre = group[(group['year'] >= shock_yr - 3) & (group['year'] < shock_yr)]
        return pre[proxy].mean()
    pm = df3.groupby('ticker').apply(_pre_mean, include_groups=False).rename(f'pre_{proxy}')
    pre_means[proxy] = pm

# Merge all pre-means
for proxy, pm in pre_means.items():
    df3 = df3.merge(pm, on='ticker', how='left')

# Event window
w3 = df3[(df3['event_time'] >= -3) & (df3['event_time'] <= 3)].copy()
w3 = w3.dropna(subset=['acsi_score', 'tech_shock_year'])
w3['post'] = (w3['event_time'] >= 0).astype(int)

# Test each proxy
for proxy in proxy_cols:
    pre_col = f'pre_{proxy}'
    w3_sub = w3.dropna(subset=[pre_col]).copy()
    if len(w3_sub) < 30:
        print(f"  {proxy}: insufficient data ({len(w3_sub)} obs), skipping")
        continue

    # Standardize
    mean_val = w3_sub[pre_col].mean()
    std_val = w3_sub[pre_col].std()
    if std_val == 0:
        print(f"  {proxy}: zero variance, skipping")
        continue
    w3_sub[f'{pre_col}_z'] = (w3_sub[pre_col] - mean_val) / std_val
    w3_sub['post_x_proxy'] = w3_sub['post'] * w3_sub[f'{pre_col}_z']

    p3 = w3_sub.set_index(['ticker', 'year'])
    n_cl3 = w3_sub['ticker'].nunique()

    mod3 = PanelOLS(p3['acsi_score'],
                    p3[['post', 'post_x_proxy']],
                    entity_effects=True, time_effects=True, check_rank=False)
    r3 = mod3.fit(cov_type='clustered', cluster_entity=True)

    print(f"\n  Proxy: {proxy}")
    print(f"  Specification: ACSI ~ Post + Post×Pre_{proxy}(z) + Firm FE + Year FE")
    print(f"  N={len(w3_sub)}, firms={n_cl3}")
    report_coef("post", r3, 'post', n_cl3)
    b, tv, p = report_coef(f"post × pre_{proxy}(z)", r3, 'post_x_proxy', n_cl3)
    if b > 0:
        print(f"    → Positive: higher pre-launch {proxy} buffers post-launch ACSI (consistent with Corollary 5)")
    else:
        print(f"    → Negative: higher pre-launch {proxy} does NOT buffer (inconsistent)")

print("\n  CAVEAT: N≈16 firms; SGA/CapEx is noisy f proxy; reverse causality possible.")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Technology Count Quadratic → ACSI (b > 1 Proxy)
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("TEST 4: FIRM-LEVEL TECHNOLOGY COUNT QUADRATIC → ACSI")
print("=" * 70)
print("Theory prediction: b > 1 implies inverted-U in firm-level technology adoption.\n")

# Merge digitalization scores with ACSI panel
digi2 = digi.copy()
digi2['company'] = digi2['brand'].map(brand_map).fillna(digi2['brand'])
merged = panel.merge(digi2[['company', 'year', 'digitalization_pct', 'cumulative_tech_adopted']],
                     on=['company', 'year'], how='left')
merged = merged.dropna(subset=['acsi_score', 'digitalization_pct'])
merged['digi2'] = merged['digitalization_pct'] ** 2
merged['tech_count'] = merged['cumulative_tech_adopted']
merged['tech_count2'] = merged['tech_count'] ** 2

n_firms_t4 = merged['company'].nunique()
n_obs_t4 = len(merged)
print(f"Merged sample: N={n_obs_t4}, firms={n_firms_t4}")
firms_list = merged['company'].unique().tolist()
print(f"Firms: {', '.join(firms_list)}")
print(f"Digi range: {merged['digitalization_pct'].min():.1f}%—{merged['digitalization_pct'].max():.1f}%")
print(f"Tech count range: {merged['tech_count'].min():.0f}—{merged['tech_count'].max():.0f}")

# --- Spec A: digitalization_pct quadratic, firm FE only ---
print(f"\n--- Spec A: ACSI ~ digi_pct + digi_pct² + Firm FE ---")
p4a = merged.set_index(['company', 'year'])
mod4a = PanelOLS(p4a['acsi_score'], p4a[['digitalization_pct', 'digi2']],
                 entity_effects=True, time_effects=False, check_rank=False)
r4a = mod4a.fit(cov_type='clustered', cluster_entity=True)
report_coef("digi_pct", r4a, 'digitalization_pct', n_firms_t4)
b2_a, t2_a, p2_a = report_coef("digi_pct²", r4a, 'digi2', n_firms_t4)
if b2_a < 0:
    tp_a = -r4a.params['digitalization_pct'] / (2 * b2_a)
    print(f"  Turning point: {tp_a:.1f}% adoption")

# --- Spec B: digitalization_pct quadratic, firm + year FE ---
print(f"\n--- Spec B: ACSI ~ digi_pct + digi_pct² + Firm FE + Year FE ---")
mod4b = PanelOLS(p4a['acsi_score'], p4a[['digitalization_pct', 'digi2']],
                 entity_effects=True, time_effects=True, check_rank=False)
r4b = mod4b.fit(cov_type='clustered', cluster_entity=True)
report_coef("digi_pct", r4b, 'digitalization_pct', n_firms_t4)
b2_b, t2_b, p2_b = report_coef("digi_pct²", r4b, 'digi2', n_firms_t4)
if b2_b < 0:
    tp_b = -r4b.params['digitalization_pct'] / (2 * b2_b)
    print(f"  Turning point: {tp_b:.1f}% adoption")
print(f"  NOTE: Year FE absorbs most variation (digi_pct is trend-like within firm)")

# --- Spec C: cumulative_tech_adopted quadratic, firm FE only ---
print(f"\n--- Spec C: ACSI ~ tech_count + tech_count² + Firm FE (robustness) ---")
mod4c = PanelOLS(p4a['acsi_score'], p4a[['tech_count', 'tech_count2']],
                 entity_effects=True, time_effects=False, check_rank=False)
r4c = mod4c.fit(cov_type='clustered', cluster_entity=True)
report_coef("tech_count", r4c, 'tech_count', n_firms_t4)
b2_c, t2_c, p2_c = report_coef("tech_count²", r4c, 'tech_count2', n_firms_t4)
if b2_c < 0:
    tp_c = -r4c.params['tech_count'] / (2 * b2_c)
    print(f"  Turning point: {tp_c:.1f} technologies adopted")

print(f"\n  CAVEAT: {n_firms_t4} firm clusters is extremely underpowered for quadratic tests.")
print(f"  Digi_pct is highly correlated with time; year FE collinearity expected.")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("SUMMARY OF EXTENDED TESTS")
print("=" * 70)
print(f"""
Test 1 (Cross-lag W→T):
  Forward: Glassdoor_{{t-1}} → ACSI_t: β={b_gd_fwd:.3f}, p={p_gd_fwd:.3f}
  Reverse: ACSI_{{t-1}} → Glassdoor_t: γ={b_acsi_rev:.4f}, p={p_acsi_rev:.3f}

Test 2 (Burnout Cliff proxy):
  Slope: δ_{{+3}} - δ_{{+1}} = {diff:.3f}, p={p_diff:.3f}
  (negative = deepening decline, consistent with slow collapse)

Test 3 (Corollary 5 proxy): see individual proxy results above

Test 4 (b > 1 proxy):
  Firm FE only: digi_pct² = {b2_a:.6f}, p={p2_a:.3f}
  Firm + Year FE: digi_pct² = {b2_b:.6f}, p={p2_b:.3f}
""")

print("[DONE] Extended tests complete.")
