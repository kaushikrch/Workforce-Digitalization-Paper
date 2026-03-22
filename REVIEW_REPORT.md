# Senior Reviewer Report — M&SOM Submission

**Paper:** "When Front-Stage Innovation Overloads Back-Stage Workers: A Dynamic Model of Workforce, Capability, and Customer Trust in Omnichannel Retail"

**Verdict:** Revise and resubmit (minor revision). The paper makes a genuine contribution — a dynamic formalisation of the service profit chain with computable thresholds — and the theory is well-executed. The remaining issues are presentation inconsistencies and a few analytical clarifications that can be addressed in a single revision round.

---

## A. Remaining Inconsistencies (Must Fix)

### A1. Inverted-U sample size mismatch (TEXT vs TABLE)

**Severity: High — contradictory claims in the same paper.**

The prose (§5.2, line ~551) states the inverted-U quadratic specification uses "N = 131, 13 firms." However, Table 4 (the regression table) correctly reports N = 216, 17 firms for specifications (2b) and (2c). The inverted-U specification requires only ACSI and e-commerce share (no CapEx or Glassdoor), so the full ACSI panel (N = 216, 17 firms) is the correct sample. The p = 0.028 was computed using t(16) degrees of freedom (G − 1 = 17 − 1 = 16), which is consistent with 17 firms, not 13.

**Fix:** Change line ~551 from "N = 131, 13 firms" to "N = 216, 17 firms" in the inverted-U paragraph. Also update the empirics introduction (§5, line ~510) where it currently says "N = 131 for the quadratic specification, 13 firms" — this should be "N = 216 for the quadratic specification, 17 firms."

### A2. Descriptive statistics table — Mean column reports Median for CapEx

**Severity: Medium.**

Table 3 (Descriptive statistics) has a column header "Mean" but the CapEx row explicitly says "CapEx / Revenue (%, median)" and reports 2.9 with "---" for SD. This is confusing: the column header promises a mean, but the row delivers a median. Either (a) report the actual mean and SD (noting the severe outliers via a footnote), (b) report median + IQR consistently with a relabeled column, or (c) add a table footnote explaining the deviation.

### A3. Abstract says "three empirical tests" but body says "three empirical strategies"

**Severity: Low.**

The abstract (line 71) uses "three empirical tests provide convergent support," while the body (§5, §5.5) consistently uses "three empirical strategies" or "three complementary identification strategies." Harmonise to a single term throughout.

### A4. Event study paragraph — AMZN exclusion not explained

**Severity: Low.**

The panel has 17 firms but the event study uses 16. The missing firm is AMZN (no omnichannel shock defined in TECH_SHOCKS). The text says "16 firms" but does not explain why one firm drops out. A parenthetical noting "(Amazon excluded: no discrete omnichannel launch event)" would be helpful.

---

## B. Analytical / Methodological Issues

### B1. Corollary 5 scope statement remains imprecise

The revision narrowed the scope ("analytically whenever the budget reaches or exceeds the Trap threshold, and computationally for B throughout [B_min/2, B_min)"), which is better. However, the Corollary as stated still begins with a condition "B ≥ B_min − ε" that covers only the near-threshold case. The computational claim for [B_min/2, B_min) is mentioned in the same sentence but has a different logical status (it's not covered by the ε-neighbourhood argument). Consider splitting into two explicit cases to match the proof structure in EC §EC.1.7.

### B2. Sensitivity table (EC Table EC.3) — σ = 1.0 row still present

The main paper now correctly states σ ∈ (1.0, 3.0] and notes "at σ = 1 the Burnout Cliff disappears." However, the sensitivity table (EC Table EC.3) still includes the row "σ: 2.0 → 1.0" and marks Theorem 1 as "Weakens" and Theorem 3 as "Weakens." This is internally consistent (σ = 1 weakens but doesn't kill Thm 1, and does kill Thm 3), but the presentation would be clearer if the σ = 1.0 row explicitly noted "Cliff disappears" rather than just "Weakens" for Theorem 3.

### B3. Within-firm Glassdoor SD claim not sourced

The paper repeatedly states "within-firm Glassdoor SD is only 30% of total SD" (lines ~549, ~553, ~644). This is a key empirical fact that justifies the between-effects estimator choice. However, the 30% figure is not verified against the actual data in the paper. [I verified it independently: actual within-firm SD / total SD ≈ 0.30, confirming the claim, but the paper should show this calculation explicitly, e.g., in the descriptive statistics or a footnote.]

### B4. Demand function — superlinear growth acknowledged but not bounded

The revision added a demand-saturation scope limitation, which is good. However, the statement "most reliable near the operating range F ∈ [F₀, 1.5 F_crit]" introduces a specific number (1.5) without justification. Why 1.5× and not 1.2× or 2×? Either cite an empirical basis or soften to "near the operating range."

---

## C. Presentation Issues

### C1. Table numbering cascade

The new regression tables (Tables 4 and 5) are added between the descriptive statistics (Table 3) and the discussion section. The paper should verify that no hardcoded "Table X" references in the text are now off by the insertion of two new tables. I did not find any hardcoded cross-references to tables beyond Table 3, but the EC references to "EC Table EC.3" and "EC Table EC.4" are hardcoded in the main paper (as noted in the production comment at line ~670). These appear correct since the EC tables are independently numbered.

### C2. County regression table — R² only for column (1)

Table 5 reports R²_within = 0.008 for column (1) but leaves it blank for columns (2)–(5). This is presumably because the R² was only saved for the main specification. Either compute and report R² for all columns, or add a footnote explaining why it is reported for (1) only.

### C3. The "Pattern 2" reference was removed — verify no orphaned references

The assessment section previously referenced "Pattern 2" which has been removed. Verify that no other section of the paper or EC still references numbered "Patterns."

### C4. EC Table EC.3 — "Notes." placeholder

Both EC Tables EC.1 (smoothing) and EC.2 (O–S mapping) have a placeholder "{Notes.}" in the table footer that contains no actual notes. Either add substantive notes or remove the placeholder.

---

## D. Publishability Assessment

### Strengths

1. **Novel contribution.** The capacity-multiplier formulation ($K_{eff} = K · h(W)$) is genuinely new in the quality-erosion literature and produces results (bistability, computable ceiling, workforce-contingent rollout speed) that no prior single model delivers.

2. **Analytical rigour.** The transparency remark (Remark 1) classifying each result as Tier A/B/C is exemplary and should be standard practice in simulation-heavy OM papers.

3. **Multi-level empirical strategy.** Testing the model's predictions at both firm level (ACSI, event study) and county level (BLS employment) is a genuine strength. The county-level test provides the statistical power that the firm-level data cannot.

4. **Honest limitations.** The paper is unusually transparent about what the data can and cannot show: the complementarity result is correctly labeled as a sign check, the COVID caveat is well-handled, and the endogeneity discussion is frank.

5. **Clean nesting.** The Oliva–Sterman special case (p = 0, a = 0) and the kill conditions in Table 2 provide clean testable restrictions.

### Weaknesses

1. **Firm-level evidence is suggestive, not conclusive.** With 13–17 firms, every firm-level result should be interpreted cautiously. The paper acknowledges this repeatedly and compensates with the county-level analysis, but the abstract and conclusion still lead with the firm-level p-values, which may overstate their evidential weight.

2. **Glassdoor as a W proxy is problematic.** Glassdoor ratings are self-selected, complaint-biased, and have low within-firm temporal variation. The paper acknowledges this (line ~644) but could benefit from a more systematic discussion of measurement error attenuation bias.

3. **No formal identification strategy for the complementarity channel.** The inverted-U and event study have plausible identification (firm FE, exogenous shock timing). The complementarity test has none — it is pure cross-sectional OLS. The paper now correctly labels this as a sign check, which is appropriate.

4. **The 37% safe-zone ratio is simulation-dependent.** The headline claim that "the neglected firm retains approximately 37% of the safe deployment headroom" comes from a specific calibration (f = 0.01 vs f = 0.10). The ratio varies with parameter choices. The paper should note this sensitivity more explicitly, perhaps reporting the range across the sensitivity grid.

### Overall

The paper makes a clear, well-structured contribution to the OM literature. The theory is the primary contribution and is strong. The empirics are honestly presented as supportive rather than definitive. The inconsistencies identified above (especially A1, the N = 131 vs N = 216 mismatch) are straightforward to fix and do not affect the substantive conclusions. After addressing the items in Section A and the higher-priority items in Sections B–C, the paper would be suitable for publication in M&SOM.

**Recommendation: Minor revision.**
