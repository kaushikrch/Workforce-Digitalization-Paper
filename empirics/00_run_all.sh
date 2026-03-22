#!/bin/bash
# WCT Empirical Analysis — Run Full Pipeline
#
# This script runs all data collection, assembly, and regression steps
# in sequence. Each step is numbered and can be run independently.
#
# Prerequisites:
#   pip install -r empirics/requirements.txt
#
# Optional: For Glassdoor sub-dimension data, download from Kaggle first:
#   kaggle datasets download -d davidgauthier/glassdoor-job-reviews
#   Then pass --input flag to step 02.
#
# Usage:
#   cd Workforce-Digitalization-Paper
#   bash empirics/00_run_all.sh

set -e  # Exit on first error

echo "=========================================="
echo " WCT EMPIRICAL PIPELINE"
echo "=========================================="
echo ""

# Step 1: ACSI data (curated from public reports)
echo "[Step 1/6] Collecting ACSI customer satisfaction data..."
python empirics/01_fetch_acsi.py
echo ""

# Step 2: Glassdoor data (from public company ratings)
echo "[Step 2/6] Processing Glassdoor employee satisfaction data..."
python empirics/02_fetch_glassdoor.py
echo ""

# Step 3: SEC EDGAR financial data (requires internet)
echo "[Step 3/6] Fetching financial data from SEC EDGAR..."
echo "  (This contacts the SEC API — may take 1-2 minutes)"
python empirics/03_fetch_edgar.py
echo ""

# Step 4: Macro controls from FRED and BLS (requires internet)
echo "[Step 4/6] Fetching macro controls from FRED/BLS..."
echo "  (This contacts FRED and BLS APIs — may take 30 seconds)"
python empirics/04_fetch_macro.py
echo ""

# Step 5: Assemble merged panel
echo "[Step 5/6] Assembling merged firm-year panel..."
python empirics/05_assemble_panel.py
echo ""

# Step 6: Run regressions
echo "[Step 6/6] Running regression analysis..."
python empirics/06_regressions.py
echo ""

echo "=========================================="
echo " PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  data/raw/acsi_scores.csv          — ACSI firm-year scores"
echo "  data/raw/glassdoor_panel.csv      — Glassdoor firm-year ratings"
echo "  data/raw/edgar_financials.csv     — SEC EDGAR financial data"
echo "  data/raw/macro_controls.csv       — FRED/BLS macro controls"
echo "  data/processed/wct_empirical_panel.csv — Merged panel"
echo "  data/processed/regression_results.csv  — Coefficient estimates"
echo "  data/processed/regression_tables.tex   — LaTeX tables"
echo ""
echo "Next steps:"
echo "  1. Review regression_results.csv for the K×W coefficient"
echo "  2. If b3 > 0 and significant → write empirical section for paper"
echo "  3. If b3 noisy → consider MSOM with suggestive evidence"
echo "  4. If b3 not detected → submit to Service Science as theory paper"
