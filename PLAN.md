# Cascade-Aware Plan: MSOM Submission-Ready Paper

## Current State Assessment

- **Document class**: `\documentclass[serv,dblanonrev]{informs4}` (Service Science — WRONG for MSOM)
- **Main paper**: ~22 pages, purely theoretical + computational study, NO empirical section
- **EC**: ~18 pages (proofs, calibration, sensitivity)
- **MSOM limits**: 32 pages main + 16 pages EC
- **Empirics**: Complete US + EU regression pipelines exist but are NOT in the paper
- **Figure issues**: 3 bugs identified (TikZ labels, burnout cliff legend, basin Fcrit label)

## MSOM Format Requirements (Key Changes Needed)

1. `\documentclass[msom,dblanonrev]{informs4}` (change from `serv`)
2. Structured abstract: **Problem definition / Methodology-results / Managerial implications**
3. Max 32 pages main body, 16 pages EC
4. Author-year citations via natbib (already in place)
5. No footnotes (subsidiary material in parentheses)
6. Code/data availability statement required

## Implementation Plan (Cascade-Ordered)

### Phase 1: MSOM Format Switch (no content dependencies)

1. Change `\documentclass[serv,...]` → `[msom,...]`
2. Restructure abstract into three subsections: Problem definition, Methodology/results, Managerial implications
3. Verify compilation still works

### Phase 2: Figure Fixes (independent of text changes)

**Figure 1 (TikZ causal loop):**
- Problem: `\textsf{R1}` etc. render as colored boxes, text invisible
- Fix: Use `text=<color>` explicitly, increase `inner sep`, use `\textbf` instead of `\textsf`, add `draw=<color>` for visible border
- Add explicit loop arrow decorations or circled labels

**Figure 2 (Basin map):**
- Problem: F_crit boundary unlabeled
- Fix: Regenerate via Python with `ax.axvline(1.11)` and text annotation "F_crit ≈ 1.11"

**Figure 5 (Burnout cliff):**
- Problem: Legend shows gray boxes for ΔF_collapse/ΔF_rapid but actual lines are colored dotted/dashed
- Fix: Regenerate via Python, match legend entries to actual line styles (colored dotted = ΔF_collapse per scenario, dashed = ΔF_rapid per scenario)

### Phase 3: Write Empirical Section (§5, after Computational Study §4)

New section: **§5. Empirical Evidence** with subsections:

**§5.1 Data and Sample**
- US panel: 13-17 retailers × 2010-2024 (ACSI, Glassdoor, SEC EDGAR, FRED)
- EU panel: 11 retailers × 2019-2023 (Trustpilot, Glassdoor, yfinance, Eurostat)
- Cite data sources: ACSI (theacsi.org), Glassdoor, SEC EDGAR XBRL API, FRED, Trustpilot, yfinance
- Report merge coverage and completeness rates
- Key variables: capex_intensity (K proxy), glassdoor_overall (W proxy), acsi_score/trustpilot_score (T proxy)

**§5.2 Identification Strategy**
- Three tests mapping to three theorems:
  - Test 1 (Complementarity → Prop 1): acsi ~ K_z + W_z + K×W_z + controls, two-way FE
  - Test 2 (Front-Stage Trap → Thm 1): acsi ~ ecom + ecom² + controls, firm FE
  - Test 3 (Good Jobs Buffer → Thm 2): Δacsi ~ ΔW + ΔW×K + controls, two-way FE
- Clustered SEs at firm level with t(G-1) correction (Cameron, Gelbach & Miller 2008)
- Small-sample corrections: HC3, CR2, wild cluster bootstrap (Webb 2023)

**§5.3 Results**
- Table 3: US regression results (4 specs for complementarity, 3 for trap, 3 for buffer)
- Table 4: EU replication results
- Honest framing:
  - K×W positive in cross-section (BE) but attenuates/reverses with firm FE
  - Inverted-U in ecommerce share (turning point ~8%)
  - Results are suggestive given G=13/11 clusters; wild bootstrap p-values reported
- Report both raw and t(G-1)-corrected p-values
- Cite CGM (2008) for small-sample correction, Webb (2023) for bootstrap

**§5.4 Cross-Geography Comparison**
- Table 5: US vs EU side-by-side (sign, magnitude, significance)
- Currency-neutral controls for EU (assets_rank)
- Trustpilot rescaled to 0-100 for comparability

### Phase 4: Update Limitations and Conclusions

**Limitations (§6.2) — add to existing:**
- Empirical limitations: few clusters (G=13/11), short Glassdoor panel, Trustpilot complaint bias
- Within-firm variation in Glassdoor is only 19-30% of total SD → FE absorbs most signal
- Fiscal year alignment: 1-3 month offsets for some EU retailers
- No store-level data (firm-level only)
- p not directly identifiable from available data (requires store-level K×W variation)

**Conclusions (§7) — add empirical paragraph:**
- Cross-sectional evidence consistent with complementarity (K×W > 0 in both geographies)
- FE estimates attenuate toward zero, consistent with low within-firm variation rather than absence of effect
- Inverted-U in ecommerce share (US) provides suggestive support for Front-Stage Trap
- Store-level panel data with exogenous technology rollouts remains the gold standard for p identification

### Phase 5: EC Updates

- Move detailed regression tables and robustness batteries to EC
- Add: §EC.5 Empirical Appendix
  - Full variable definitions and data sources
  - Merge diagnostics and balance checks
  - All robustness specifications (lagged, COVID-excluded, leave-one-out, wild bootstrap)
  - EU-specific: fiscal year alignment, currency handling, Trustpilot scaling

### Phase 6: Senior Reviewer Audit

Check for:
1. Notation consistency (K, W, T, F, G, S, D) across main, EC, and figures
2. Theorem numbering matches between main and EC
3. Cross-references: EC §EC.X references accurate after restructuring
4. Parameter values consistent (b=1.3, σ=2.0, etc.) across text, tables, figures
5. Claim-evidence alignment: every empirical claim backed by specific table/column
6. No orphaned references or undefined terms
7. Accessibility: jargon explained on first use, acronyms defined
8. Mathematical depth: proofs stay in EC, only theorem statements + intuition in main

### Phase 7: Length Management

Target: 32 pages main, 16 pages EC
- Current main: ~22 pages → adding empirical section (~6 pages) = ~28 pages ✓
- If over 32 pages:
  - Cut: lengthy proof sketches in main (move to EC)
  - Compress: Related Literature (currently verbose)
  - Compress: Computational Study narration
  - Move detailed parameter calibration to EC
- EC currently ~18 pages → need to trim to 16
  - Compress sensitivity tables
  - Remove redundant notation sections

### Phase 8: Final Deliverables

Generate ZIP containing:
```
submission/
├── WCT_Paper_MSOM.tex          # Main paper (MSOM format)
├── WCT_Paper_MSOM.pdf          # Compiled PDF
├── WCT_EC_MSOM.tex             # Electronic companion
├── WCT_EC_MSOM.pdf             # Compiled EC PDF
├── informs4.cls                # Template class
├── figures/
│   ├── fig2_basin_map.pdf
│   ├── fig3_complementarity.pdf
│   ├── fig4_hidden_cost.pdf
│   └── fig5_burnout_cliff.pdf
├── replication/
│   ├── README.md               # Replication instructions
│   ├── requirements.txt        # Python dependencies
│   ├── wct_simulation.py       # Simulation code
│   ├── empirics/
│   │   ├── 01_fetch_acsi.py
│   │   ├── 02_fetch_glassdoor.py
│   │   ├── 03_fetch_edgar.py
│   │   ├── 04_fetch_macro.py
│   │   ├── 05_assemble_panel.py
│   │   ├── 06_regressions.py
│   │   ├── 07_alternative_specs.py
│   │   ├── 08_robustness.py
│   │   ├── eu_01_fetch_all_data.py
│   │   └── eu_02_assemble_and_test.py
│   └── data/
│       ├── raw/                # Input CSVs
│       └── processed/          # Output panels + results
└── references.bib              # Bibliography
```

## Dependency Graph (Cascade Order)

```
Phase 1 (format) ──┐
Phase 2 (figures) ──┼──→ Phase 6 (audit) ──→ Phase 7 (trim) ──→ Phase 8 (deliverables)
Phase 3 (empirics) ─┤
Phase 4 (lim/conc) ─┤
Phase 5 (EC) ───────┘
```

Phases 1-2 are independent. Phases 3-5 have mild dependencies (empirics → limitations → EC).
Phase 6 must follow all content changes. Phase 7-8 are final sequential steps.
