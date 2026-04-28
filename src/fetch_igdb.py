import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

CLIENT_ID     = os.getenv("IGDB_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("IGDB_CLIENT_SECRET", "")
INPUT_FILE    = "data/pc_catalog.csv"
OUTPUT_PATH   = Path("data/master_games.csv")

PLATFORMS = {
    "Game Boy Color":  22,
    "Game Boy Advance":24,
    "GameCube":        21,
    "Nintendo DS":     20,
    "Wii":              5,
    "Wii U":           41,
    "Nintendo 3DS":    37,
}

ESRB_MAP      = {1: "RP", 2: "EC", 3: "E", 4: "E10+", 5: "T", 6: "M", 7: "AO"}
PAGE_SIZE     = 500
DELAY         = 0.26
MATCH_THRESHOLD = 0.6

STOP_WORDS = {"the", "a", "an", "of", "and", "in", "for", "version", "edition"}

CONSOLE_YEAR_RANGE = {
    22: (1998, 2003),
    24: (2001, 2008),
    21: (2001, 2007),
    20: (2004, 2013),
     5: (2006, 2013),
    41: (2012, 2017),
    37: (2011, 2020),
}


def normalize(title: str) -> str:
    t = unicodedata.normalize("NFD", str(title)).encode("ascii", errors="ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return " ".join(t.split())


def key_words(title: str) -> set[str]:
    return {w for w in normalize(title).split() if w not in STOP_WORDS}


def similarity(a: str, b: str) -> float:
    ka, kb = key_words(a), key_words(b)
    if not ka:
        return 1.0
    return len(ka & kb) / len(ka)


def best_igdb_match(pc_title: str, igdb_games: list[dict], platform_id: int | None = None) -> dict | None:
    if not igdb_games:
        return None
    scored = [(similarity(pc_title, g.get("name", "")), g) for g in igdb_games]
    best_score = max(s for s, _ in scored)
    if best_score < MATCH_THRESHOLD:
        return None
    candidates = [g for s, g in scored if s >= best_score - 0.05]
    year_lo, year_hi = CONSOLE_YEAR_RANGE.get(platform_id, (1900, 2100)) if platform_id else (1900, 2100)

    def rank(g):
        is_main = 0 if g.get("category", 99) == 0 else 1
        rds = g.get("release_dates") or []
        years = [
            datetime.utcfromtimestamp(rd["date"]).year
            for rd in rds
            if rd.get("date") and rd.get("platform") == platform_id
        ]
        year = min(years) if years else None
        in_range = 0 if (year and year_lo <= year <= year_hi) else 1
        return (is_main, in_range, g.get("id") or 999999)

    return min(candidates, key=rank)


def get_token() -> str:
    r = requests.post(
        "https://id.twitch.tv/oauth2/token",
        params={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "grant_type": "client_credentials"},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def igdb_post(endpoint: str, query: str, headers: dict) -> list:
    r = requests.post(f"https://api.igdb.com/v4/{endpoint}", headers=headers, data=query)
    r.raise_for_status()
    return r.json()


def fetch_all_igdb(platform_id: int, headers: dict) -> list[dict]:
    query = (
        "fields id, name, category, genres.name, involved_companies.company.name, "
        "involved_companies.publisher, age_ratings.organization, age_ratings.rating_category, "
        "release_dates.date, release_dates.region, release_dates.platform, "
        "rating, aggregated_rating, aggregated_rating_count, platforms, "
        "remakes, remasters, ports, franchises.name; "
        f"where platforms = ({platform_id});"
    )
    results, offset = [], 0
    while True:
        page = igdb_post("games", f"{query} limit {PAGE_SIZE}; offset {offset};", headers)
        if not page:
            break
        results.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(DELAY)
    return results


def fetch_rerelease_details(ids: list[int], headers: dict) -> dict:
    results = {}
    for i in range(0, len(ids), PAGE_SIZE):
        batch = ids[i:i + PAGE_SIZE]
        page = igdb_post(
            "games",
            f"fields name, aggregated_rating, rating, platforms, release_dates.date, release_dates.region; "
            f"where id = ({','.join(map(str, batch))}); limit {PAGE_SIZE};",
            headers,
        )
        for g in page:
            results[g["id"]] = g
        time.sleep(DELAY)
    return results


def extract_publisher(involved_companies):
    for c in (involved_companies or []):
        if c.get("publisher") and c.get("company"):
            return c["company"].get("name")
    first = (involved_companies or [None])[0]
    return first["company"].get("name") if first and first.get("company") else None


def extract_esrb(age_ratings):
    for r in (age_ratings or []):
        if r.get("organization") == 1:
            return ESRB_MAP.get(r.get("rating_category"))
    return None


def extract_release_year(release_dates, platform_id):
    if not release_dates:
        return None
    na = [rd for rd in release_dates if rd.get("platform") == platform_id and rd.get("region") == 2]
    candidates = na or [rd for rd in release_dates if rd.get("platform") == platform_id]
    ts = candidates[0].get("date") if candidates else None
    return datetime.utcfromtimestamp(ts).year if ts else None


def extract_rerelease_ids(game):
    ids = []
    for field in ("remakes", "remasters", "ports"):
        ids.extend(game.get(field) or [])
    return list(set(ids))


def extract_rerelease_type(game) -> str | None:
    if game.get("remakes"):
        return "remake"
    if game.get("remasters"):
        return "remaster"
    if game.get("ports"):
        return "port"
    return None


def _earliest_release_year(release_dates) -> int | None:
    years = []
    for rd in (release_dates or []):
        ts = rd.get("date")
        if ts:
            try:
                years.append(datetime.utcfromtimestamp(ts).year)
            except Exception:
                pass
    return min(years) if years else None


def _is_valid_rerelease(entry: dict, platform_id: int) -> bool:
    rds = entry.get("release_dates") or []
    has_us = any(rd.get("region") == 2 and rd.get("date") for rd in rds)
    if not has_us:
        return False
    plats = set(entry.get("platforms") or [])
    # Exclude if exclusively on the same platform as the original
    if plats and plats <= {platform_id}:
        return False
    return True


def build_rerelease_row(details, igdb_game, platform_id):
    valid = {rid: g for rid, g in details.items() if _is_valid_rerelease(g, platform_id)}
    if not valid:
        return {
            "rerelease_exists": False, "rerelease_type": None, "rerelease_year": None,
            "rerelease_critic_score": None, "rerelease_user_score": None,
            "rerelease_critic_user_delta": None, "rerelease_igdb_ids": None,
        }
    primary = max(valid.values(), key=lambda g: g.get("aggregated_rating_count") or 0)
    critic, user = primary.get("aggregated_rating"), primary.get("rating")
    year = _earliest_release_year(primary.get("release_dates"))
    return {
        "rerelease_exists":            True,
        "rerelease_type":              extract_rerelease_type(igdb_game),
        "rerelease_year":              year,
        "rerelease_critic_score":      critic,
        "rerelease_user_score":        user,
        "rerelease_critic_user_delta": round(critic - user, 2) if critic and user else None,
        "rerelease_igdb_ids":          "|".join(str(x) for x in valid.keys()),
    }


def main():
    catalog = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(catalog)} games from {INPUT_FILE}")
    print(catalog.groupby("console")["pc_title"].count().to_string())
    print()

    token = get_token()
    headers = {"Client-ID": CLIENT_ID, "Authorization": f"Bearer {token}"}

    rows = []
    for console_name, platform_id in PLATFORMS.items():
        pc_games = catalog[catalog["console"] == console_name]
        if pc_games.empty:
            print(f"{console_name}: no PC games found — skipping")
            continue

        print(f"{console_name}: fetching {len(pc_games)} PC games from IGDB...")
        igdb_games = fetch_all_igdb(platform_id, headers)
        print(f"  IGDB returned {len(igdb_games)} games for platform {platform_id}")

        igdb_by_norm: dict[str, list] = {}
        for g in igdb_games:
            key = normalize(g.get("name", ""))
            igdb_by_norm.setdefault(key, []).append(g)

        matched_igdb = []
        pc_row_data = []

        matched = unmatched = 0
        for _, pc_row in pc_games.iterrows():
            pc_title = pc_row["pc_title"]
            norm_pc  = normalize(pc_title)

            candidates = igdb_by_norm.get(norm_pc, igdb_games)
            igdb_game  = best_igdb_match(pc_title, candidates, platform_id)

            if igdb_game:
                matched += 1
                matched_igdb.append(igdb_game)
            else:
                unmatched += 1

            pc_row_data.append((pc_row, igdb_game))

        print(f"  Matched: {matched} | Unmatched (null IGDB fields): {unmatched}")

        all_rr_ids = set()
        for g in matched_igdb:
            all_rr_ids.update(extract_rerelease_ids(g))
        rr_lookup = fetch_rerelease_details(list(all_rr_ids), headers) if all_rr_ids else {}

        for pc_row, igdb_game in pc_row_data:
            if igdb_game:
                rr_ids = extract_rerelease_ids(igdb_game)
                rr_data = {rid: rr_lookup[rid] for rid in rr_ids if rid in rr_lookup}
                critic  = igdb_game.get("aggregated_rating")
                user    = igdb_game.get("rating")
                rows.append({
                    "igdb_id":                 igdb_game.get("id"),
                    "title":                   pc_row["pc_title"],
                    "console":                 console_name,
                    "pc_url":                  pc_row["pc_url"],
                    "genre":                   "|".join(g["name"] for g in (igdb_game.get("genres") or []) if "name" in g) or None,
                    "publisher":               extract_publisher(igdb_game.get("involved_companies")),
                    "esrb":                    extract_esrb(igdb_game.get("age_ratings")),
                    "release_year":            extract_release_year(igdb_game.get("release_dates", []), platform_id),
                    "franchise_igdb_ids":      "|".join(str(f["id"]) for f in (igdb_game.get("franchises") or []) if isinstance(f, dict)),
                    "franchise_names":         "|".join(f["name"] for f in (igdb_game.get("franchises") or []) if isinstance(f, dict) and f.get("name")),
                    "igdb_critic_score":       critic,
                    "igdb_user_score":         user,
                    "igdb_rating_count":       igdb_game.get("aggregated_rating_count"),
                    "critic_user_score_delta": round(critic - user, 2) if critic and user else None,
                    "igdb_category":           igdb_game.get("category", 0),
                    **build_rerelease_row(rr_data, igdb_game, platform_id),
                })
            else:
                rows.append({
                    "igdb_id":                 None,
                    "title":                   pc_row["pc_title"],
                    "console":                 console_name,
                    "pc_url":                  pc_row["pc_url"],
                    "genre":                   None, "publisher": None, "esrb": None,
                    "release_year":            None, "franchise_igdb_ids": None, "franchise_names": None,
                    "igdb_critic_score":       None, "igdb_user_score": None,
                    "igdb_rating_count":       None, "critic_user_score_delta": None,
                    "igdb_category":           None,
                    "rerelease_exists":        False, "rerelease_type": None, "rerelease_year": None,
                    "rerelease_critic_score":  None, "rerelease_user_score": None,
                    "rerelease_critic_user_delta": None, "rerelease_igdb_ids": None,
                })

        print()

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["pc_url"], inplace=True)
    df.sort_values(["console", "title"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{len(df)} games saved to {OUTPUT_PATH}")
    print(df.groupby("console")["title"].count().to_string())
    match_rate = df["igdb_id"].notna().mean()
    print(f"IGDB match rate: {match_rate:.1%}")


if __name__ == "__main__":
    main()
