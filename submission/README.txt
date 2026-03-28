=========================================================================
When Front-Stage Innovation Overloads Back-Stage Workers:
A Dynamic Model of Workforce, Capability, and Customer Trust
in Omnichannel Retail

Submission to Manufacturing & Service Operations Management (M&SOM)
=========================================================================

CONTENTS
--------

1. Main Paper
   WCT_Paper_INFORMS.pdf      Main manuscript (32 pages)

2. Electronic Companion
   WCT_EC_INFORMS.pdf         Full proofs, sensitivity, calibration,
                               empirical appendix (16 pages)

3. LaTeX Source
   WCT_Paper_INFORMS.tex      Main paper source
   WCT_EC_INFORMS.tex         EC source
   informs4.cls               INFORMS document class
   eqndefns-left.sty          Equation formatting
   fig2_basin_map.pdf          Figure 2: Basin-of-attraction map
   fig3_complementarity.pdf    Figure 3: Complementarity
   fig4_hidden_cost.pdf        Figure 4: Hidden cost of app innovation
   fig5_burnout_cliff.pdf      Figure 5: Burnout cliff

4. Simulation Code
   wct_simulation.py           ODE simulation and figure generation
   Requirements: numpy, scipy, matplotlib

5. Empirical Analysis Code
   empirics/
     01_fetch_acsi.py          ACSI data collection
     02_fetch_glassdoor.py     Glassdoor data collection
     03_fetch_edgar.py         SEC EDGAR financials
     03b_fetch_tech_invest.py  EDGAR technology investment
     04_fetch_macro.py         FRED/BLS macro controls
     05_assemble_panel.py      Panel assembly (v1 and v2)
     06_regressions.py         Core firm-level regressions
     07_alternative_specs.py   Alternative specifications
     08_robustness.py          Robustness checks (bootstrap, CR3, HAC)
     09_event_study.py         Omnichannel launch event study
     10_fetch_store_data.py    Store location data (OpenStreetMap)
     11_county_regressions.py  County-level Good Jobs Buffer
     12_firm_digitalization.py Firm-specific digitalization tests
     13_state_minwage_buffer.py State minimum wage instrument test
     14_cbp_establishment.py   Establishment-margin buffer test
     eu_01_fetch_all_data.py   European panel data collection
     eu_02_assemble_and_test.py European panel analysis
     requirements.txt          Python dependencies (pinned)
   Requirements: pandas, numpy, scipy, statsmodels, linearmodels

6. Data
   data/raw/                   Raw data from ACSI, Glassdoor, EDGAR, FRED
   data/processed/             Assembled panels (v1: 25 cols, v2: 34 cols)
   data/bls_qcew/              BLS county-level retail employment/wages
   data/store_level/           Technology adoption timelines, store locations
   data/shift_project/         Shift Project codebook (data via Harvard Dataverse)


DATA SOURCES
------------

Firm-level:
  - ACSI scores: theacsi.org (2008-2024, 17 firms)
  - Glassdoor ratings: glassdoor.com (2010-2023, 16 firms)
  - Financial data: SEC EDGAR XBRL API (2008-2024, 17 firms)
  - E-commerce share: FRED series ECOMPCTSA

County-level:
  - BLS QCEW: Quarterly Census of Employment and Wages
    NAICS 44-45 (retail trade), 2015-2024
    3,115 counties, 30,951 county-year observations
  - State minimum wages: DOL/EPI (2015-2024, 53 jurisdictions)

European (exploratory):
  - Trustpilot, Glassdoor, Yahoo Finance, UK ONS, Eurostat
    11 retailers, 2019-2023


REPLICATION
-----------

To replicate all empirical results:

  pip install -r empirics/requirements.txt
  cd empirics
  python 01_fetch_acsi.py
  python 02_fetch_glassdoor.py
  python 03_fetch_edgar.py
  python 03b_fetch_tech_invest.py
  python 04_fetch_macro.py
  python 05_assemble_panel.py
  python 06_regressions.py
  python 09_event_study.py
  python 11_county_regressions.py
  python 13_state_minwage_buffer.py
  python 14_cbp_establishment.py

To regenerate simulation figures:

  python wct_simulation.py

To compile LaTeX:

  pdflatex WCT_Paper_INFORMS.tex
  pdflatex WCT_Paper_INFORMS.tex
  pdflatex WCT_EC_INFORMS.tex
  pdflatex WCT_EC_INFORMS.tex


CONTACT
-------

Corresponding author information suppressed for double-blind review.
