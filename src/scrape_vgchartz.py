import re
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

DATA_DIR    = Path(__file__).parent.parent / "data"
MASTER_PATH = DATA_DIR / "master_games.csv"
OUTPUT_PATH = DATA_DIR / "sales_data.csv"
CACHE_PATH  = DATA_DIR / "vgchartz_raw_cache.csv"

BASE_URL         = "https://www.vgchartz.com"
RESULTS_PER_PAGE = 200
REQUEST_DELAY    = 1.5
FUZZY_THRESHOLD  = 85

CONSOLE_SLUGS = {
    "Game Boy Color":  "GBC",
    "Game Boy Advance": "GBA",
    "GameCube":        "GC",
    "Nintendo DS":     "DS",
    "Wii":             "Wii",
    "Wii U":           "WiiU",
    "Nintendo 3DS":    "3DS",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def normalize(title):
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", title.lower())).strip()


def parse_sales(text):
    if not text or not isinstance(text, str):
        return None
    t = text.strip().lower()
    if t in ("n/a", "", "—", "-", "<0.01"):
        return None
    m = re.search(r"([\d.]+)m", t)
    return float(m.group(1)) if m else None


def list_url(slug, page):
    return (
        f"{BASE_URL}/games/games.php?console={requests.utils.quote(slug)}"
        f"&ownership=Both&results={RESULTS_PER_PAGE}&page={page}"
        f"&order=Name&showtotalsales=1&showpublisher=1&showvgchartzscore=0"
        f"&shownasales=1&showdeveloper=0&showreleasedate=1&showlastupdate=0&showgenre=0"
    )


def parse_page(html, console):
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("div", id="generalBody")
    if not body:
        return []
    rows = []
    for tr in body.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 8:
            continue
        link = tds[2].find("a", href=True)
        if not link or "/game/" not in link.get("href", ""):
            continue
        href = link["href"]
        if not href.startswith("http"):
            href = BASE_URL + href
        title = link.get_text(strip=True)
        if not title:
            continue
        rows.append({
            "vgchartz_title":      title,
            "vgchartz_url":        href,
            "console":             console,
            "units_sold_global_m": parse_sales(tds[5].get_text(strip=True)),
            "units_sold_na_m":     parse_sales(tds[6].get_text(strip=True)),
        })
    return rows


def scrape_console(console, slug, session):
    all_rows, page = [], 1
    while True:
        try:
            r = session.get(list_url(slug, page), headers=HTTP_HEADERS, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error on {console} page {page}: {e}")
            break
        rows = parse_page(r.text, console)
        if not rows:
            break
        all_rows.extend(rows)
        print(f"  {console} page {page}: {len(rows)} games")
        if len(rows) < RESULTS_PER_PAGE:
            break
        page += 1
        time.sleep(REQUEST_DELAY)
    return all_rows


def match_titles(master_df, vgc_df):
    results = []
    for console, group in master_df.groupby("console"):
        subset = vgc_df[vgc_df["console"] == console].copy()
        if subset.empty:
            for _, row in group.iterrows():
                results.append({**row.to_dict(), "vgchartz_title": None, "vgchartz_url": None,
                                 "units_sold_global_m": None, "units_sold_na_m": None,
                                 "match_type": "unmatched", "match_score": None})
            continue

        subset["_norm"] = subset["vgchartz_title"].apply(normalize)
        subset = subset.drop_duplicates("_norm").reset_index(drop=True)
        norm_map  = dict(zip(subset["_norm"], subset.index))
        norm_list = list(subset["_norm"])
        exact = fuzzy = unmatched = 0

        for _, master_row in group.iterrows():
            norm = normalize(master_row["title"])
            if norm in norm_map:
                vgc = subset.loc[norm_map[norm]]
                results.append({**master_row.to_dict(), "vgchartz_title": vgc["vgchartz_title"],
                                 "vgchartz_url": vgc["vgchartz_url"],
                                 "units_sold_global_m": vgc["units_sold_global_m"],
                                 "units_sold_na_m": vgc["units_sold_na_m"],
                                 "match_type": "exact", "match_score": 100.0})
                exact += 1
            else:
                best = process.extractOne(norm, norm_list, scorer=fuzz.WRatio, score_cutoff=FUZZY_THRESHOLD)
                if best:
                    _, score, idx = best
                    vgc = subset.iloc[idx]
                    results.append({**master_row.to_dict(), "vgchartz_title": vgc["vgchartz_title"],
                                    "vgchartz_url": vgc["vgchartz_url"],
                                    "units_sold_global_m": vgc["units_sold_global_m"],
                                    "units_sold_na_m": vgc["units_sold_na_m"],
                                    "match_type": "fuzzy", "match_score": round(score, 1)})
                    fuzzy += 1
                else:
                    results.append({**master_row.to_dict(), "vgchartz_title": None, "vgchartz_url": None,
                                    "units_sold_global_m": None, "units_sold_na_m": None,
                                    "match_type": "unmatched", "match_score": None})
                    unmatched += 1

        print(f"  {console}: {exact} exact  {fuzzy} fuzzy  {unmatched} unmatched")
    return pd.DataFrame(results)


def main():
    master_df = pd.read_csv(MASTER_PATH)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    if CACHE_PATH.exists():
        print(f"Loading cached VGChartz data...")
        vgc_df = pd.read_csv(CACHE_PATH)
    else:
        print("Scraping VGChartz...")
        rows = []
        for console, slug in CONSOLE_SLUGS.items():
            rows.extend(scrape_console(console, slug, session))
            time.sleep(REQUEST_DELAY)
        vgc_df = pd.DataFrame(rows).drop_duplicates(subset=["vgchartz_title", "console"]).reset_index(drop=True)
        vgc_df.to_csv(CACHE_PATH, index=False)
        print(f"Scraped {len(vgc_df)} VGChartz entries")

    print("\nMatching titles...")
    result_df = match_titles(master_df, vgc_df)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{len(result_df)} rows saved to {OUTPUT_PATH}")
    print(result_df["match_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
