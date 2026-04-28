import io
import json
import os
import re
import sys
import time

import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT_FILE  = "data/pc_catalog.csv"
OUTPUT_FILE = "data/price_history.csv"
RATE_LIMIT  = 1.5

CONDITION_MAP = {
    "used":       "loose",
    "cib":        "cib",
    "new":        "new",
    "graded":     "graded",
    "boxonly":    "boxonly",
    "manualonly": "manualonly",
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})


def extract_chart_data(html: str) -> dict | None:
    match = re.search(r"VGPC\.chart_data\s*=\s*(\{.*?\});", html, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def parse_price_history(igdb_id, title: str, console: str, chart_data: dict) -> list[dict]:
    rows = []
    for vgpc_cond, friendly_cond in CONDITION_MAP.items():
        series = chart_data.get(vgpc_cond)
        if not series:
            continue
        for entry in series:
            if len(entry) != 2:
                continue
            ts_ms, price_cents = entry
            if price_cents is None or price_cents <= 0:
                continue
            rows.append({
                "igdb_id":   igdb_id,
                "title":     title,
                "console":   console,
                "condition": friendly_cond,
                "date":      pd.to_datetime(ts_ms, unit="ms").strftime("%Y-%m-%d"),
                "price_usd": round(price_cents / 100, 2),
            })
    return rows


def load_completed_urls() -> set:
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        return set(pd.read_csv(OUTPUT_FILE, usecols=["pc_url"])["pc_url"].dropna().unique())
    except Exception:
        return set()


def append_rows(rows: list[dict], write_header: bool) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)


def main():
    games = pd.read_csv(INPUT_FILE, usecols=["pc_title", "console", "pc_url"])
    games = games.rename(columns={"pc_title": "title"})
    games["igdb_id"] = None
    games = games.dropna(subset=["pc_url"])
    total = len(games)
    print(f"Loaded {total} games from {INPUT_FILE}")

    completed_urls = load_completed_urls()
    print(f"Already scraped: {len(completed_urls)} — resuming from checkpoint")

    first_write = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    scraped = skipped = failed = 0

    for i, row in games.iterrows():
        pc_url  = row["pc_url"]
        title   = row["title"]
        console = row["console"]
        igdb_id = row["igdb_id"]

        if pc_url in completed_urls:
            skipped += 1
            continue

        resp = SESSION.get(pc_url, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            print(f"  [{i+1}/{total}] FETCH ERROR — {title} ({console})")
            append_rows([{"igdb_id": igdb_id, "title": title, "console": console,
                          "pc_url": pc_url, "condition": "fetch_error", "date": None, "price_usd": None}],
                        first_write)
            first_write = False
            failed += 1
            time.sleep(RATE_LIMIT)
            continue

        chart_data = extract_chart_data(resp.text)
        if chart_data is None:
            print(f"  [{i+1}/{total}] NO CHART DATA — {title} ({console})")
            append_rows([{"igdb_id": igdb_id, "title": title, "console": console,
                          "pc_url": pc_url, "condition": "no_data", "date": None, "price_usd": None}],
                        first_write)
            first_write = False
            failed += 1
            time.sleep(RATE_LIMIT)
            continue

        rows = parse_price_history(igdb_id, title, console, chart_data)
        if rows:
            for r in rows:
                r["pc_url"] = pc_url
            append_rows(rows, first_write)
            first_write = False
            scraped += 1
            print(f"  [{i+1}/{total}] OK ({len(rows)} records) — {title} ({console})")
        else:
            print(f"  [{i+1}/{total}] EMPTY — {title} ({console})")
            append_rows([{"igdb_id": igdb_id, "title": title, "console": console,
                          "pc_url": pc_url, "condition": "empty", "date": None, "price_usd": None}],
                        first_write)
            first_write = False
            failed += 1

        time.sleep(RATE_LIMIT)

    print(f"\nDone. Scraped: {scraped} | Skipped: {skipped} | Failed: {failed} | Total: {total}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
