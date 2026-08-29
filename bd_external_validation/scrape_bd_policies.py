"""
Fetches raw HTML and extracts main-body text for the 35-company BD subset
(data/raw/bd_privacy_data_subset_35.csv), per PROJECT_CONTEXT.md §3.4 Phase B/C.

Writes:
  bd_external_validation/raw_html/<No>_<slug>.html   -- raw response body, for audit/replay
  bd_external_validation/data/bd_extracted_text.csv  -- one row per company: status + extracted text

Run:
    .venv/bin/python bd_external_validation/scrape_bd_policies.py
"""
import csv
import os
import re
import time

import requests
from bs4 import BeautifulSoup

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET_PATH = os.path.join(REPO_ROOT, "data", "raw", "bd_privacy_data_subset_35.csv")
RAW_HTML_DIR = os.path.join(REPO_ROOT, "bd_external_validation", "raw_html")
OUT_PATH = os.path.join(REPO_ROOT, "bd_external_validation", "data", "bd_extracted_text.csv")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 15
STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "svg", "form", "iframe"]


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


def extract_main_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: s.strip().startswith("<!--")):
        comment.extract()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def fetch_one(url, retries=1):
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text, None
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2)
    return None, last_error


def main():
    os.makedirs(RAW_HTML_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(SUBSET_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    ok, failed = 0, 0
    for r in rows:
        no = r["No."]
        name = r["Company Name"].strip()
        url = r["Privacy Policy URL"].strip()
        slug = slugify(name)

        print(f"[{no:>3}] {name} -> {url}", flush=True)
        html, error = fetch_one(url)

        if html is None:
            print(f"      FAILED: {error}", flush=True)
            failed += 1
            results.append({
                "No.": no, "Company Name": name, "Industry": r["Industry"],
                "Privacy Policy URL": url, "status": "failed", "error": error,
                "char_count": 0, "extracted_text": "",
            })
            time.sleep(1)
            continue

        html_path = os.path.join(RAW_HTML_DIR, f"{no}_{slug}.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        extracted = extract_main_text(html)
        print(f"      OK ({len(extracted):,} chars extracted)", flush=True)
        ok += 1
        results.append({
            "No.": no, "Company Name": name, "Industry": r["Industry"],
            "Privacy Policy URL": url, "status": "ok", "error": "",
            "char_count": len(extracted), "extracted_text": extracted,
        })
        time.sleep(1)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone: {ok} ok, {failed} failed, out of {len(rows)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
