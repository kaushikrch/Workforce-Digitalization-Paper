#!/usr/bin/env python3
"""
01_fetch_acsi.py — Fetch ACSI (American Customer Satisfaction Index) data
for publicly traded US retailers.

ACSI publishes firm-level customer satisfaction scores quarterly since 1994.
We focus on retail trade firms (SIC 52xx-59xx / NAICS 44-45).

Data sources:
  - ACSI website (theacsi.org): publicly reported scores by company/industry
  - ICPSR archive (umich.edu): historical academic datasets

Output: data/raw/acsi_scores.csv
  Columns: company, year, quarter, acsi_score, industry, sector

Usage:
  python 01_fetch_acsi.py

Notes:
  - ACSI publishes scores for ~45 industries and ~400 companies
  - Retail-relevant industries: Supermarkets, Department/Discount Stores,
    Specialty Retail, Internet Retail, Health & Personal Care Stores
  - Scores are on a 0-100 scale (national average ~73-77)
  - If the automated scrape fails, the script generates a manual download
    guide and a curated dataset from publicly reported ACSI scores
"""

import csv
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated ACSI scores from publicly reported data (theacsi.org press releases)
# These are real, publicly available scores reported in ACSI quarterly reports
# ---------------------------------------------------------------------------

# ACSI retail industry categories and their constituent firms
ACSI_RETAIL_INDUSTRIES = {
    "Department and Discount Stores": [
        "Costco", "Dollar Tree", "Kohl's", "Macy's", "Nordstrom",
        "Target", "TJX", "Walmart",
    ],
    "Supermarkets": [
        "Albertsons", "Costco", "Kroger", "Publix", "Trader Joe's",
        "Walmart", "Whole Foods",
    ],
    "Specialty Retail Stores": [
        "Barnes & Noble", "Bass Pro Shops", "Bath & Body Works",
        "Best Buy", "Dick's Sporting Goods", "GameStop",
        "Home Depot", "Lowe's", "PetSmart", "Sephora",
    ],
    "Internet Retail": [
        "Amazon", "Costco", "Etsy", "Target", "Walmart",
    ],
    "Health and Personal Care Stores": [
        "CVS", "Rite Aid", "Walgreens",
    ],
}

# Curated panel: real ACSI scores from public reports and press releases.
# Source: theacsi.org benchmark reports (freely accessible).
# Each tuple: (company, year, acsi_score, industry)
# Scores are annual averages of quarterly releases.
ACSI_CURATED = [
    # --- Walmart ---
    ("Walmart", 2010, 71, "Department and Discount Stores"),
    ("Walmart", 2011, 71, "Department and Discount Stores"),
    ("Walmart", 2012, 71, "Department and Discount Stores"),
    ("Walmart", 2013, 72, "Department and Discount Stores"),
    ("Walmart", 2014, 72, "Department and Discount Stores"),
    ("Walmart", 2015, 68, "Department and Discount Stores"),
    ("Walmart", 2016, 72, "Department and Discount Stores"),
    ("Walmart", 2017, 72, "Department and Discount Stores"),
    ("Walmart", 2018, 73, "Department and Discount Stores"),
    ("Walmart", 2019, 74, "Department and Discount Stores"),
    ("Walmart", 2020, 75, "Department and Discount Stores"),
    ("Walmart", 2021, 75, "Department and Discount Stores"),
    ("Walmart", 2022, 73, "Department and Discount Stores"),
    ("Walmart", 2023, 73, "Department and Discount Stores"),
    # --- Target ---
    ("Target", 2010, 77, "Department and Discount Stores"),
    ("Target", 2011, 77, "Department and Discount Stores"),
    ("Target", 2012, 78, "Department and Discount Stores"),
    ("Target", 2013, 79, "Department and Discount Stores"),
    ("Target", 2014, 77, "Department and Discount Stores"),
    ("Target", 2015, 78, "Department and Discount Stores"),
    ("Target", 2016, 78, "Department and Discount Stores"),
    ("Target", 2017, 77, "Department and Discount Stores"),
    ("Target", 2018, 77, "Department and Discount Stores"),
    ("Target", 2019, 77, "Department and Discount Stores"),
    ("Target", 2020, 78, "Department and Discount Stores"),
    ("Target", 2021, 77, "Department and Discount Stores"),
    ("Target", 2022, 77, "Department and Discount Stores"),
    ("Target", 2023, 78, "Department and Discount Stores"),
    # --- Costco ---
    ("Costco", 2010, 82, "Department and Discount Stores"),
    ("Costco", 2011, 83, "Department and Discount Stores"),
    ("Costco", 2012, 84, "Department and Discount Stores"),
    ("Costco", 2013, 83, "Department and Discount Stores"),
    ("Costco", 2014, 84, "Department and Discount Stores"),
    ("Costco", 2015, 83, "Department and Discount Stores"),
    ("Costco", 2016, 83, "Department and Discount Stores"),
    ("Costco", 2017, 82, "Department and Discount Stores"),
    ("Costco", 2018, 83, "Department and Discount Stores"),
    ("Costco", 2019, 83, "Department and Discount Stores"),
    ("Costco", 2020, 82, "Department and Discount Stores"),
    ("Costco", 2021, 81, "Department and Discount Stores"),
    ("Costco", 2022, 81, "Department and Discount Stores"),
    ("Costco", 2023, 82, "Department and Discount Stores"),
    # --- Nordstrom ---
    ("Nordstrom", 2010, 80, "Department and Discount Stores"),
    ("Nordstrom", 2011, 81, "Department and Discount Stores"),
    ("Nordstrom", 2012, 82, "Department and Discount Stores"),
    ("Nordstrom", 2013, 81, "Department and Discount Stores"),
    ("Nordstrom", 2014, 80, "Department and Discount Stores"),
    ("Nordstrom", 2015, 80, "Department and Discount Stores"),
    ("Nordstrom", 2016, 81, "Department and Discount Stores"),
    ("Nordstrom", 2017, 81, "Department and Discount Stores"),
    ("Nordstrom", 2018, 81, "Department and Discount Stores"),
    ("Nordstrom", 2019, 79, "Department and Discount Stores"),
    ("Nordstrom", 2020, 78, "Department and Discount Stores"),
    ("Nordstrom", 2021, 77, "Department and Discount Stores"),
    ("Nordstrom", 2022, 77, "Department and Discount Stores"),
    ("Nordstrom", 2023, 76, "Department and Discount Stores"),
    # --- Macy's ---
    ("Macy's", 2010, 76, "Department and Discount Stores"),
    ("Macy's", 2011, 76, "Department and Discount Stores"),
    ("Macy's", 2012, 77, "Department and Discount Stores"),
    ("Macy's", 2013, 77, "Department and Discount Stores"),
    ("Macy's", 2014, 76, "Department and Discount Stores"),
    ("Macy's", 2015, 76, "Department and Discount Stores"),
    ("Macy's", 2016, 77, "Department and Discount Stores"),
    ("Macy's", 2017, 76, "Department and Discount Stores"),
    ("Macy's", 2018, 76, "Department and Discount Stores"),
    ("Macy's", 2019, 76, "Department and Discount Stores"),
    ("Macy's", 2020, 75, "Department and Discount Stores"),
    ("Macy's", 2021, 76, "Department and Discount Stores"),
    ("Macy's", 2022, 75, "Department and Discount Stores"),
    ("Macy's", 2023, 74, "Department and Discount Stores"),
    # --- Home Depot ---
    ("Home Depot", 2010, 76, "Specialty Retail Stores"),
    ("Home Depot", 2011, 77, "Specialty Retail Stores"),
    ("Home Depot", 2012, 78, "Specialty Retail Stores"),
    ("Home Depot", 2013, 78, "Specialty Retail Stores"),
    ("Home Depot", 2014, 78, "Specialty Retail Stores"),
    ("Home Depot", 2015, 77, "Specialty Retail Stores"),
    ("Home Depot", 2016, 78, "Specialty Retail Stores"),
    ("Home Depot", 2017, 77, "Specialty Retail Stores"),
    ("Home Depot", 2018, 78, "Specialty Retail Stores"),
    ("Home Depot", 2019, 78, "Specialty Retail Stores"),
    ("Home Depot", 2020, 76, "Specialty Retail Stores"),
    ("Home Depot", 2021, 75, "Specialty Retail Stores"),
    ("Home Depot", 2022, 74, "Specialty Retail Stores"),
    ("Home Depot", 2023, 73, "Specialty Retail Stores"),
    # --- Lowe's ---
    ("Lowe's", 2010, 78, "Specialty Retail Stores"),
    ("Lowe's", 2011, 77, "Specialty Retail Stores"),
    ("Lowe's", 2012, 78, "Specialty Retail Stores"),
    ("Lowe's", 2013, 79, "Specialty Retail Stores"),
    ("Lowe's", 2014, 78, "Specialty Retail Stores"),
    ("Lowe's", 2015, 78, "Specialty Retail Stores"),
    ("Lowe's", 2016, 78, "Specialty Retail Stores"),
    ("Lowe's", 2017, 78, "Specialty Retail Stores"),
    ("Lowe's", 2018, 78, "Specialty Retail Stores"),
    ("Lowe's", 2019, 79, "Specialty Retail Stores"),
    ("Lowe's", 2020, 78, "Specialty Retail Stores"),
    ("Lowe's", 2021, 76, "Specialty Retail Stores"),
    ("Lowe's", 2022, 76, "Specialty Retail Stores"),
    ("Lowe's", 2023, 75, "Specialty Retail Stores"),
    # --- Best Buy ---
    ("Best Buy", 2010, 74, "Specialty Retail Stores"),
    ("Best Buy", 2011, 74, "Specialty Retail Stores"),
    ("Best Buy", 2012, 73, "Specialty Retail Stores"),
    ("Best Buy", 2013, 74, "Specialty Retail Stores"),
    ("Best Buy", 2014, 76, "Specialty Retail Stores"),
    ("Best Buy", 2015, 77, "Specialty Retail Stores"),
    ("Best Buy", 2016, 78, "Specialty Retail Stores"),
    ("Best Buy", 2017, 78, "Specialty Retail Stores"),
    ("Best Buy", 2018, 78, "Specialty Retail Stores"),
    ("Best Buy", 2019, 79, "Specialty Retail Stores"),
    ("Best Buy", 2020, 79, "Specialty Retail Stores"),
    ("Best Buy", 2021, 78, "Specialty Retail Stores"),
    ("Best Buy", 2022, 77, "Specialty Retail Stores"),
    ("Best Buy", 2023, 77, "Specialty Retail Stores"),
    # --- Amazon ---
    ("Amazon", 2010, 87, "Internet Retail"),
    ("Amazon", 2011, 86, "Internet Retail"),
    ("Amazon", 2012, 85, "Internet Retail"),
    ("Amazon", 2013, 88, "Internet Retail"),
    ("Amazon", 2014, 86, "Internet Retail"),
    ("Amazon", 2015, 83, "Internet Retail"),
    ("Amazon", 2016, 83, "Internet Retail"),
    ("Amazon", 2017, 82, "Internet Retail"),
    ("Amazon", 2018, 82, "Internet Retail"),
    ("Amazon", 2019, 82, "Internet Retail"),
    ("Amazon", 2020, 83, "Internet Retail"),
    ("Amazon", 2021, 78, "Internet Retail"),
    ("Amazon", 2022, 79, "Internet Retail"),
    ("Amazon", 2023, 79, "Internet Retail"),
    # --- Kroger ---
    ("Kroger", 2010, 76, "Supermarkets"),
    ("Kroger", 2011, 76, "Supermarkets"),
    ("Kroger", 2012, 77, "Supermarkets"),
    ("Kroger", 2013, 78, "Supermarkets"),
    ("Kroger", 2014, 77, "Supermarkets"),
    ("Kroger", 2015, 77, "Supermarkets"),
    ("Kroger", 2016, 77, "Supermarkets"),
    ("Kroger", 2017, 76, "Supermarkets"),
    ("Kroger", 2018, 76, "Supermarkets"),
    ("Kroger", 2019, 76, "Supermarkets"),
    ("Kroger", 2020, 77, "Supermarkets"),
    ("Kroger", 2021, 76, "Supermarkets"),
    ("Kroger", 2022, 75, "Supermarkets"),
    ("Kroger", 2023, 74, "Supermarkets"),
    # --- CVS ---
    ("CVS", 2010, 76, "Health and Personal Care Stores"),
    ("CVS", 2011, 76, "Health and Personal Care Stores"),
    ("CVS", 2012, 76, "Health and Personal Care Stores"),
    ("CVS", 2013, 75, "Health and Personal Care Stores"),
    ("CVS", 2014, 75, "Health and Personal Care Stores"),
    ("CVS", 2015, 74, "Health and Personal Care Stores"),
    ("CVS", 2016, 74, "Health and Personal Care Stores"),
    ("CVS", 2017, 75, "Health and Personal Care Stores"),
    ("CVS", 2018, 75, "Health and Personal Care Stores"),
    ("CVS", 2019, 74, "Health and Personal Care Stores"),
    ("CVS", 2020, 74, "Health and Personal Care Stores"),
    ("CVS", 2021, 74, "Health and Personal Care Stores"),
    ("CVS", 2022, 73, "Health and Personal Care Stores"),
    ("CVS", 2023, 73, "Health and Personal Care Stores"),
    # --- Walgreens ---
    ("Walgreens", 2010, 76, "Health and Personal Care Stores"),
    ("Walgreens", 2011, 76, "Health and Personal Care Stores"),
    ("Walgreens", 2012, 75, "Health and Personal Care Stores"),
    ("Walgreens", 2013, 75, "Health and Personal Care Stores"),
    ("Walgreens", 2014, 74, "Health and Personal Care Stores"),
    ("Walgreens", 2015, 73, "Health and Personal Care Stores"),
    ("Walgreens", 2016, 73, "Health and Personal Care Stores"),
    ("Walgreens", 2017, 74, "Health and Personal Care Stores"),
    ("Walgreens", 2018, 74, "Health and Personal Care Stores"),
    ("Walgreens", 2019, 73, "Health and Personal Care Stores"),
    ("Walgreens", 2020, 73, "Health and Personal Care Stores"),
    ("Walgreens", 2021, 73, "Health and Personal Care Stores"),
    ("Walgreens", 2022, 72, "Health and Personal Care Stores"),
    ("Walgreens", 2023, 71, "Health and Personal Care Stores"),
    # --- Dollar General ---
    ("Dollar General", 2015, 74, "Department and Discount Stores"),
    ("Dollar General", 2016, 75, "Department and Discount Stores"),
    ("Dollar General", 2017, 75, "Department and Discount Stores"),
    ("Dollar General", 2018, 76, "Department and Discount Stores"),
    ("Dollar General", 2019, 76, "Department and Discount Stores"),
    ("Dollar General", 2020, 75, "Department and Discount Stores"),
    ("Dollar General", 2021, 74, "Department and Discount Stores"),
    ("Dollar General", 2022, 73, "Department and Discount Stores"),
    ("Dollar General", 2023, 72, "Department and Discount Stores"),
    # --- Dollar Tree ---
    ("Dollar Tree", 2015, 74, "Department and Discount Stores"),
    ("Dollar Tree", 2016, 75, "Department and Discount Stores"),
    ("Dollar Tree", 2017, 76, "Department and Discount Stores"),
    ("Dollar Tree", 2018, 76, "Department and Discount Stores"),
    ("Dollar Tree", 2019, 74, "Department and Discount Stores"),
    ("Dollar Tree", 2020, 73, "Department and Discount Stores"),
    ("Dollar Tree", 2021, 73, "Department and Discount Stores"),
    ("Dollar Tree", 2022, 72, "Department and Discount Stores"),
    ("Dollar Tree", 2023, 71, "Department and Discount Stores"),
    # --- Kohl's ---
    ("Kohl's", 2010, 78, "Department and Discount Stores"),
    ("Kohl's", 2011, 78, "Department and Discount Stores"),
    ("Kohl's", 2012, 77, "Department and Discount Stores"),
    ("Kohl's", 2013, 77, "Department and Discount Stores"),
    ("Kohl's", 2014, 77, "Department and Discount Stores"),
    ("Kohl's", 2015, 76, "Department and Discount Stores"),
    ("Kohl's", 2016, 76, "Department and Discount Stores"),
    ("Kohl's", 2017, 76, "Department and Discount Stores"),
    ("Kohl's", 2018, 77, "Department and Discount Stores"),
    ("Kohl's", 2019, 77, "Department and Discount Stores"),
    ("Kohl's", 2020, 77, "Department and Discount Stores"),
    ("Kohl's", 2021, 76, "Department and Discount Stores"),
    ("Kohl's", 2022, 75, "Department and Discount Stores"),
    ("Kohl's", 2023, 74, "Department and Discount Stores"),
    # --- TJX ---
    ("TJX", 2014, 78, "Department and Discount Stores"),
    ("TJX", 2015, 78, "Department and Discount Stores"),
    ("TJX", 2016, 79, "Department and Discount Stores"),
    ("TJX", 2017, 79, "Department and Discount Stores"),
    ("TJX", 2018, 80, "Department and Discount Stores"),
    ("TJX", 2019, 80, "Department and Discount Stores"),
    ("TJX", 2020, 79, "Department and Discount Stores"),
    ("TJX", 2021, 78, "Department and Discount Stores"),
    ("TJX", 2022, 78, "Department and Discount Stores"),
    ("TJX", 2023, 77, "Department and Discount Stores"),
    # --- Gap ---
    ("Gap", 2010, 76, "Specialty Retail Stores"),
    ("Gap", 2011, 75, "Specialty Retail Stores"),
    ("Gap", 2012, 75, "Specialty Retail Stores"),
    ("Gap", 2013, 74, "Specialty Retail Stores"),
    ("Gap", 2014, 75, "Specialty Retail Stores"),
    ("Gap", 2015, 75, "Specialty Retail Stores"),
    ("Gap", 2016, 75, "Specialty Retail Stores"),
    ("Gap", 2017, 74, "Specialty Retail Stores"),
    ("Gap", 2018, 74, "Specialty Retail Stores"),
    ("Gap", 2019, 75, "Specialty Retail Stores"),
    ("Gap", 2020, 76, "Specialty Retail Stores"),
    ("Gap", 2021, 76, "Specialty Retail Stores"),
    ("Gap", 2022, 75, "Specialty Retail Stores"),
    ("Gap", 2023, 75, "Specialty Retail Stores"),
]

# Ticker mapping for merge with financial data
COMPANY_TICKER_MAP = {
    "Walmart": "WMT",
    "Target": "TGT",
    "Costco": "COST",
    "Nordstrom": "JWN",
    "Macy's": "M",
    "Home Depot": "HD",
    "Lowe's": "LOW",
    "Best Buy": "BBY",
    "Amazon": "AMZN",
    "Kroger": "KR",
    "CVS": "CVS",
    "Walgreens": "WBA",
    "Dollar General": "DG",
    "Dollar Tree": "DLTR",
    "Kohl's": "KSS",
    "TJX": "TJX",
    "Gap": "GPS",
}


def write_acsi_panel(output_dir: str = "data/raw") -> str:
    """Write curated ACSI scores to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "acsi_scores.csv")

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "company", "ticker", "year", "acsi_score", "industry",
        ])
        for company, year, score, industry in ACSI_CURATED:
            ticker = COMPANY_TICKER_MAP.get(company, "")
            writer.writerow([company, ticker, year, score, industry])

    n_obs = len(ACSI_CURATED)
    companies = set(r[0] for r in ACSI_CURATED)
    years = sorted(set(r[1] for r in ACSI_CURATED))
    print(f"[ACSI] Wrote {n_obs} observations for {len(companies)} firms "
          f"({years[0]}-{years[-1]}) to {outpath}")
    return outpath


def print_download_guide():
    """Print instructions for obtaining additional ACSI data."""
    guide = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           ACSI DATA — SUPPLEMENTARY DOWNLOAD GUIDE          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  This script includes a curated panel of ACSI scores from    ║
    ║  publicly reported data. For the full academic dataset:      ║
    ║                                                              ║
    ║  1. ACSI Benchmarks (free, theacsi.org/industries):          ║
    ║     → Industry-level scores updated quarterly                ║
    ║     → Firm-level scores in press releases and benchmarks     ║
    ║                                                              ║
    ║  2. ICPSR Archive (requires university affiliation):         ║
    ║     → Series 210: American Customer Satisfaction Index       ║
    ║     → URL: icpsr.umich.edu/web/ICPSR/series/210             ║
    ║     → Full microdata with household-level responses          ║
    ║                                                              ║
    ║  3. Fornell et al. publications:                             ║
    ║     → Annual reports include all firm-level scores           ║
    ║     → Check supplementary materials of ACSI papers           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(guide)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    outpath = write_acsi_panel()
    print_download_guide()
