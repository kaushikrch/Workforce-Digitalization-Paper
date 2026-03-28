#!/usr/bin/env python3
"""
03b_fetch_tech_invest.py — Fetch digitalization-specific investment data
from SEC EDGAR for retail firms.

Variables extracted:
  - CapitalizedComputerSoftwareGross: cumulative software asset (K_digital)
  - CapitalizedComputerSoftwareAdditions: annual software investment (I_digital)
  - IntangibleAssetsNetExcludingGoodwill: broader tech/IP proxy
  - ResearchAndDevelopmentExpense: R&D spending (minimal for retailers)
  - EmployeeRelatedLiabilitiesCurrent: wage/benefits liability (W_financial)

Output: data/raw/tech_invest.csv
"""

import csv
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SEC_USER_AGENT = "WCT-Research academic-research@university.edu"

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

# Tags grouped by what they capture
TECH_TAGS = {
    "cap_software_gross": [
        "CapitalizedComputerSoftwareGross",
    ],
    "cap_software_net": [
        "CapitalizedComputerSoftwareNet",
    ],
    "cap_software_additions": [
        "CapitalizedComputerSoftwareAdditions",
        "CapitalizedComputerSoftwarePeriodIncreaseDecrease",
    ],
    "cap_software_amort": [
        "CapitalizedComputerSoftwareAmortization",
        "CapitalizedComputerSoftwareAmortization1",
    ],
    "intangibles_net": [
        "IntangibleAssetsNetExcludingGoodwill",
    ],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "TechnologyAndContentExpense",
    ],
    "emp_liab_current": [
        "EmployeeRelatedLiabilitiesCurrent",
    ],
    "emp_liab_total": [
        "EmployeeRelatedLiabilitiesCurrentAndNoncurrent",
    ],
}


def extract_annual(facts, tag_list, taxonomy="us-gaap"):
    """Extract annual 10-K values for a list of XBRL tags."""
    results = {}
    tax_data = facts.get("facts", {}).get(taxonomy, {})
    for tag in tag_list:
        tag_data = tax_data.get(tag, {})
        for unit_key in ("USD", "pure"):
            entries = tag_data.get("units", {}).get(unit_key, [])
            for e in entries:
                if e.get("form") not in ("10-K", "10-K/A"):
                    continue
                end = e.get("end", "")
                if len(end) >= 4:
                    try:
                        fy = int(end[:4])
                    except ValueError:
                        continue
                    if 2008 <= fy <= 2025 and fy not in results:
                        results[fy] = e.get("val")
    return results


def fetch_all(output_dir="data/raw"):
    """Fetch tech investment + employee liability data for all firms."""
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for ticker, (company, cik) in RETAIL_FIRMS.items():
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        req = Request(url)
        req.add_header("User-Agent", SEC_USER_AGENT)
        try:
            with urlopen(req, timeout=30) as resp:
                facts = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError) as e:
            print(f"  [WARN] {company}: {e}")
            time.sleep(0.15)
            continue

        # Extract all tag groups
        data = {}
        for var_name, tags in TECH_TAGS.items():
            data[var_name] = extract_annual(facts, tags)

        # Also get revenue for computing intensities
        rev_tags = ["Revenues", "SalesRevenueNet",
                    "RevenueFromContractWithCustomerExcludingAssessedTax"]
        data["revenue"] = extract_annual(facts, rev_tags)

        # Collect all years
        all_years = set()
        for d in data.values():
            all_years.update(d.keys())

        for year in sorted(all_years):
            rev = data["revenue"].get(year)
            row = {
                "company": company,
                "ticker": ticker,
                "year": year,
                "cap_software_gross": data["cap_software_gross"].get(year, ""),
                "cap_software_net": data["cap_software_net"].get(year, ""),
                "cap_software_additions": data["cap_software_additions"].get(year, ""),
                "cap_software_amort": data["cap_software_amort"].get(year, ""),
                "intangibles_net": data["intangibles_net"].get(year, ""),
                "rd_expense": data["rd_expense"].get(year, ""),
                "emp_liab_current": data["emp_liab_current"].get(year, ""),
                "emp_liab_total": data["emp_liab_total"].get(year, ""),
                "revenue": rev or "",
            }
            # Compute intensities
            for var in ["cap_software_gross", "intangibles_net",
                        "rd_expense", "emp_liab_current"]:
                val = row[var]
                if val and rev:
                    try:
                        row[f"{var}_rev"] = round(float(val) / float(rev), 6)
                    except (ValueError, ZeroDivisionError):
                        row[f"{var}_rev"] = ""
                else:
                    row[f"{var}_rev"] = ""

            all_rows.append(row)

        n_sw = len(data["cap_software_gross"])
        n_int = len(data["intangibles_net"])
        n_emp = len(data["emp_liab_current"])
        print(f"  {ticker:6s} ({company:15s}): software={n_sw:>2}, "
              f"intangibles={n_int:>2}, emp_liab={n_emp:>2} years")
        time.sleep(0.15)

    # Write
    outpath = os.path.join(output_dir, "tech_invest.csv")
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda r: (r["ticker"], r["year"])))

    print(f"\n[TECH] Wrote {len(all_rows)} obs to {outpath}")
    # Coverage summary
    n_sw = sum(1 for r in all_rows if r.get("cap_software_gross"))
    n_int = sum(1 for r in all_rows if r.get("intangibles_net"))
    n_emp = sum(1 for r in all_rows if r.get("emp_liab_current"))
    print(f"[TECH] Coverage: software={n_sw}, intangibles={n_int}, "
          f"emp_liab={n_emp} firm-years")
    return outpath


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    fetch_all()
