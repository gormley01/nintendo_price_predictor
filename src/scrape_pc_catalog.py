import io
import os
import sys
import time

import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUTPUT_FILE = "data/pc_catalog.csv"
RATE_LIMIT  = 1.5

CONSOLES = {
    "Game Boy Color":  "gameboy-color",
    "Game Boy Advance":"gameboy-advance",
    "GameCube":        "gamecube",
    "Nintendo DS":     "nintendo-ds",
    "Wii":             "wii",
    "Wii U":           "wii-u",
    "Nintendo 3DS":    "nintendo-3ds",
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


def scrape_console(console_name: str, console_slug: str) -> list[dict]:
    games  = []
    cursor = 0
    while True:
        url  = (
            f"https://www.pricecharting.com/console/{console_slug}"
            f"?q=&sort=title&dir=asc&status=&cursor={cursor}&format=json"
        )
        resp = SESSION.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  [{console_name}] cursor={cursor}: HTTP {resp.status_code} — stopping")
            break

        data     = resp.json()
        products = data.get("products", [])
        if not products:
            break

        for p in products:
            games.append({
                "pc_title": p["productName"],
                "console":  console_name,
                "pc_url":   f"https://www.pricecharting.com/game/{p['consoleUri']}/{p['productUri']}",
            })

        next_cursor = data.get("cursor")
        print(f"  [{console_name}] cursor={cursor}: {len(products)} games (total: {len(games)})")

        if next_cursor is None or next_cursor == cursor or len(products) < 150:
            break
        cursor = next_cursor
        time.sleep(RATE_LIMIT)

    return games


def main():
    if os.path.exists(OUTPUT_FILE):
        completed = set(pd.read_csv(OUTPUT_FILE)["console"].unique())
        print(f"Resuming — already scraped: {completed}")
    else:
        completed = set()

    first_write = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    total = 0

    for console_name, console_slug in CONSOLES.items():
        if console_name in completed:
            print(f"[{console_name}] already done — skipping")
            continue

        print(f"\nScraping {console_name}...")
        games = scrape_console(console_name, console_slug)
        if games:
            df = pd.DataFrame(games).drop_duplicates(subset=["pc_url"])
            df.to_csv(OUTPUT_FILE, mode="a", header=first_write, index=False)
            first_write = False
            total += len(df)
            print(f"  → {len(df)} unique games saved")
        else:
            print(f"  → no games found")

        time.sleep(RATE_LIMIT)

    print(f"\nDone. Total games in catalog: {total}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
