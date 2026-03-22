REPLICATION PACKAGE
===================
"When Front-Stage Innovation Overloads Back-Stage Workers:
 A Dynamic Model of Workforce, Capability, and Customer Trust
 in Omnichannel Retail"

Submitted to M&SOM (Manufacturing & Service Operations Management)


CONTENTS
--------
submission/
  WCT_Paper_INFORMS.tex       Main paper (MSOM format, 25 pages)
  WCT_Paper_INFORMS.pdf       Compiled main paper
  WCT_EC_INFORMS.tex          Electronic Companion (15 pages)
  WCT_EC_INFORMS.pdf          Compiled EC
  informs4.cls                INFORMS document class
  *.sty                       Required style files
  fig2_basin_map.pdf           Figure 2: Basin-of-attraction map
  fig3_complementarity.pdf     Figure 3: Digitalisation-workforce complementarity
  fig4_hidden_cost.pdf         Figure 4: Hidden cost of app innovation
  fig5_burnout_cliff.pdf       Figure 5: Burnout cliff
  fig_study3_roi_decomp.pdf    EC figure: ROI decomposition
  informs_Logo.pdf             Template logo

simulation/
  wct_simulation.py            Full simulation code (generates all figures)
  Requirements: numpy, scipy, matplotlib
  Usage: python wct_simulation.py

empirics/
  US pipeline (run in order):
    01_fetch_acsi.py            Fetch ACSI scores
    02_fetch_glassdoor.py       Fetch Glassdoor ratings
    03_fetch_edgar.py           Fetch SEC EDGAR financials
    03b_fetch_tech_invest.py    Fetch technology investment data
    04_fetch_macro.py           Fetch macro controls (FRED, BLS)
    05_assemble_panel.py        Merge into firm-year panel
    06_regressions.py           Core regression tests (3 tests)
    07_alternative_specs.py     Alternative specifications
    08_robustness.py            Full robustness battery

  EU pipeline:
    eu_01_fetch_all_data.py     Fetch all EU data sources
    eu_02_assemble_and_test.py  Assemble panel + run all tests

  Requirements: pandas, numpy, statsmodels, linearmodels, scipy, yfinance

data/
  raw/                          Input data files
  processed/                    Generated panels and regression results


REPRODUCTION STEPS
------------------
1. Simulation figures:
   cd simulation && python wct_simulation.py

2. US empirical results:
   cd empirics
   python 01_fetch_acsi.py        # Requires internet
   python 02_fetch_glassdoor.py   # Requires internet
   python 03_fetch_edgar.py       # Requires internet
   python 04_fetch_macro.py       # Requires internet
   python 05_assemble_panel.py
   python 06_regressions.py
   python 07_alternative_specs.py
   python 08_robustness.py

3. EU empirical results:
   python eu_01_fetch_all_data.py # Requires internet
   python eu_02_assemble_and_test.py

4. LaTeX compilation:
   pdflatex WCT_Paper_INFORMS.tex  (run twice for cross-references)
   pdflatex WCT_EC_INFORMS.tex     (run twice)


KEY METHODOLOGICAL NOTES
------------------------
- All regressions use t(G-1) corrected p-values (Cameron, Gelbach & Miller 2008)
- Wild cluster bootstrap (Webb 6-point) reported for G < 20 clusters
- EU between-effects use currency-neutral controls (asset percentile rank)
- Merge validation: 1:1 key uniqueness enforced, duplicate detection logged
- Fiscal year alignment documented for non-calendar-year EU firms
