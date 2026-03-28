#!/usr/bin/env python3
"""
03_fetch_edgar.py — Fetch financial data from SEC EDGAR for retail firms.

Replaces Compustat/WRDS for our purposes. Uses the SEC's free EDGAR
full-text search and XBRL API to pull key financial variables from
10-K annual reports.

Variables extracted:
  - Revenue (Revenues / SalesRevenueNet)
  - Capital Expenditure (PaymentsToAcquirePropertyPlantAndEquipment)
  - SGA Expense (SellingGeneralAndAdministrativeExpense)
  - Total Assets (Assets)
  - Number of Employees (EntityNumberOfEmployees)
  - Net Income (NetIncomeLoss)

Output: data/raw/edgar_financials.csv
  Columns: company, ticker, cik, year, revenue, capex, sga, assets,
           employees, net_income, capex_intensity, sga_intensity

Usage:
  python 03_fetch_edgar.py

Notes:
  - SEC EDGAR requires a User-Agent header with contact info
  - Rate limit: max 10 requests/second (we use 0.15s delay)
  - XBRL company facts API: data.sec.gov/api/xbrl/companyfacts/
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SEC requires a user-agent with contact information
SEC_USER_AGENT = "WCT-Research academic-research@university.edu"

# Rate limit: SEC allows 10 req/sec; we stay well below
REQUEST_DELAY = 0.2  # seconds between requests

# Target firms: ticker -> (company_name, CIK number)
# CIK numbers from SEC EDGAR company search
RETAIL_FIRMS = {
    "WMT":  ("Walmart",        104169),
    "TGT":  ("Target",          27419),
    "COST": ("Costco",         909832),
    "JWN":  ("Nordstrom",       72333),
    "M":    ("Macy's",         794367),
    "HD":   ("Home Depot",     354950),
    "LOW":  ("Lowe's",          60667),
    "BBY":  ("Best Buy",       764478),
    "AMZN": ("Amazon",        1018724),
    "KR":   ("Kroger",          56873),
    "CVS":  ("CVS",             64803),
    "WBA":  ("Walgreens",     1618921),
    "DG":   ("Dollar General",  34067),
    "DLTR": ("Dollar Tree",    935703),
    "KSS":  ("Kohl's",         885639),
    "TJX":  ("TJX",            109198),
    "GPS":  ("Gap",             39911),
}

# XBRL tags to extract
XBRL_TAGS = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditureDiscontinuedOperations",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
    ],
    "assets": [
        "Assets",
    ],
    "employees": [
        "EntityNumberOfEmployees",
    ],
    "net_income": [
        "NetIncomeLoss",
    ],
}


def fetch_company_facts(cik: int) -> dict | None:
    """Fetch XBRL company facts from SEC EDGAR API."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    req = Request(url)
    req.add_header("User-Agent", SEC_USER_AGENT)
    req.add_header("Accept", "application/json")

    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"  [WARN] HTTP {e.code} for CIK {cik}: {e.reason}")
        return None
    except URLError as e:
        print(f"  [WARN] URL error for CIK {cik}: {e.reason}")
        return None


def extract_annual_values(facts: dict, tag_list: list,
                          start_year: int = 2008,
                          end_year: int = 2024) -> dict:
    """
    Extract annual values for a given XBRL tag from company facts.

    Returns: {year: value} dict
    """
    results = {}

    # Search both us-gaap and dei taxonomies
    for taxonomy in ("us-gaap", "dei"):
        tax_data = facts.get("facts", {}).get(taxonomy, {})
        for tag in tag_list:
            tag_data = tax_data.get(tag, {})
            units = tag_data.get("units", {})
            # Try USD first, then pure (for employee counts)
            for unit_key in ("USD", "pure", "shares"):
                entries = units.get(unit_key, [])
                for entry in entries:
                    # Only annual filings (10-K)
                    form = entry.get("form", "")
                    if form not in ("10-K", "10-K/A"):
                        continue
                    # Extract fiscal year
                    end_date = entry.get("end", "")
                    if len(end_date) >= 4:
                        try:
                            fy = int(end_date[:4])
                        except ValueError:
                            continue
                        if start_year <= fy <= end_year:
                            val = entry.get("val")
                            if val is not None and fy not in results:
                                results[fy] = val

    return results


def fetch_all_firms(output_dir: str = "data/raw") -> str:
    """Fetch financial data for all target retail firms."""
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "edgar_financials.csv")

    all_rows = []
    n_firms = len(RETAIL_FIRMS)

    for i, (ticker, (company, cik)) in enumerate(RETAIL_FIRMS.items(), 1):
        print(f"[EDGAR] ({i}/{n_firms}) Fetching {company} "
              f"(CIK {cik}, ticker {ticker})...")

        facts = fetch_company_facts(cik)
        if facts is None:
            print(f"  [SKIP] Could not retrieve data for {company}")
            time.sleep(REQUEST_DELAY)
            continue

        # Extract each variable
        data = {}
        for var_name, tags in XBRL_TAGS.items():
            data[var_name] = extract_annual_values(facts, tags)

        # Determine year range from revenue (most complete series)
        years = sorted(data.get("revenue", {}).keys())
        if not years:
            print(f"  [SKIP] No revenue data found for {company}")
            time.sleep(REQUEST_DELAY)
            continue

        for year in years:
            rev = data["revenue"].get(year)
            capex = data["capex"].get(year)
            sga = data["sga"].get(year)
            assets = data["assets"].get(year)
            empl = data["employees"].get(year)
            ni = data["net_income"].get(year)

            # Compute intensity ratios
            capex_intensity = round(capex / rev, 4) if (capex and rev) else ""
            sga_intensity = round(sga / rev, 4) if (sga and rev) else ""

            all_rows.append({
                "company": company,
                "ticker": ticker,
                "cik": cik,
                "year": year,
                "revenue": rev or "",
                "capex": capex or "",
                "sga": sga or "",
                "assets": assets or "",
                "employees": empl or "",
                "net_income": ni or "",
                "capex_intensity": capex_intensity,
                "sga_intensity": sga_intensity,
            })

        print(f"  [OK] {len(years)} years of data ({years[0]}-{years[-1]})")
        time.sleep(REQUEST_DELAY)

    # Write output
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "company", "ticker", "cik", "year",
            "revenue", "capex", "sga", "assets",
            "employees", "net_income",
            "capex_intensity", "sga_intensity",
        ])
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda r: (r["ticker"], r["year"])))

    firms_ok = len(set(r["ticker"] for r in all_rows))
    print(f"\n[EDGAR] Wrote {len(all_rows)} firm-year observations "
          f"for {firms_ok} firms to {outpath}")
    return outpath


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    fetch_all_firms()
