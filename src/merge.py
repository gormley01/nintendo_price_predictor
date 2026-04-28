import re
import ast

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

INPUT_MASTER         = DATA_DIR / "master_games.csv"
INPUT_SALES          = DATA_DIR / "sales_data.csv"
INPUT_PRICES         = DATA_DIR / "price_history.csv"
OUTPUT_MERGED        = DATA_DIR / "merged_dataset.csv"
OUTPUT_EXCLUDED      = DATA_DIR / "excluded_games.csv"

MIN_PRICE_RECORDS = 50

BUDGET_REPRINT_KEYWORDS  = ["player's choice", "players choice", "best seller",
                             "nintendo selects", "greatest hits", "platinum"]
RARE_VARIANT_KEYWORDS    = ["not for resale", "limited edition", "special edition",
                             "collector's edition", "collectors edition", "1st print",
                             "foil box", "for rental", "kiosk", "demo disc",
                             "inaugural edition", "launch edition", "world edition"]
RETAILER_EXCLUSIVE_KW    = ["kmart", "walmart", "toys r us", "target edition",
                             "best buy", "eb games", "gamestop exclusive"]
BUNDLE_KEYWORDS          = ["with dvd", "bundle", "combo", "pack", "twin pack"]

def classify_variant(title: str) -> tuple[bool, str]:
    t = title.lower()
    if any(k in t for k in BUDGET_REPRINT_KEYWORDS):
        return True, "budget_reprint"
    if any(k in t for k in RARE_VARIANT_KEYWORDS):
        return True, "rare_variant"
    if any(k in t for k in RETAILER_EXCLUSIVE_KW):
        return True, "retailer_exclusive"
    if any(k in t for k in BUNDLE_KEYWORDS):
        return True, "bundle"
    if re.search(r"\[.+\]", title):
        return True, "variant"
    return False, "standard"

CONDITION_BY_CONSOLE = {
    "Game Boy Color":   "loose",
    "Game Boy Advance": "loose",
    "GameCube":         "cib",
    "Nintendo DS":      "cib",
    "Wii":              "cib",
    "Wii U":            "cib",
    "Nintendo 3DS":     "cib",
}

def build_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    sentinel_conditions = {"not_found", "no_data", "empty", "fetch_error"}
    price_df = price_df[~price_df["condition"].isin(sentinel_conditions)].copy()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    price_df = price_df.dropna(subset=["date", "price_usd", "pc_url"])

    rows = []
    for pc_url, group in price_df.groupby("pc_url"):
        console   = group["console"].iloc[0]
        canonical = CONDITION_BY_CONSOLE.get(console)
        canon_data = group[group["condition"] == canonical].sort_values("date")

        record_count = len(canon_data)
        history = dict(zip(
            canon_data["date"].dt.strftime("%Y-%m-%d"),
            canon_data["price_usd"].round(2),
        ))
        rows.append({
            "pc_url":              pc_url,
            "price_record_count":  record_count,
            "price_history_json":  str(history),
            "canonical_condition": canonical,
        })
    return pd.DataFrame(rows)

def deduplicate_price_histories(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "price_history_json" not in df.columns:
        return df, pd.DataFrame()

    has_hist  = df["price_history_json"].notna() & (df["price_record_count"].fillna(0) >= MIN_PRICE_RECORDS)
    with_hist = df[has_hist].copy()
    without   = df[~has_hist]

    hist_counts = with_hist["price_history_json"].value_counts()
    dup_hists   = hist_counts[hist_counts > 1].index

    if len(dup_hists) == 0:
        return df, pd.DataFrame()

    match_rank = {"exact": 0, "fuzzy": 1, "unmatched": 2, None: 3}
    with_hist["_match_rank"]  = with_hist["match_type"].map(match_rank).fillna(3)
    with_hist["_match_score"] = pd.to_numeric(with_hist.get("match_score", 0), errors="coerce").fillna(0)
    with_hist["_title_len"]   = with_hist["title"].str.len()

    keep_ids = set()
    dup_ids  = set()

    for hist in dup_hists:
        group = with_hist[with_hist["price_history_json"] == hist].sort_values(
            ["igdb_id", "_match_rank", "_match_score", "_title_len"],
            ascending=[True, True, False, True],
        )
        keep_ids.add(group.iloc[0]["igdb_id"])
        dup_ids.update(group.iloc[1:]["igdb_id"].tolist())

    unique_hist_ids = set(with_hist[~with_hist["price_history_json"].isin(dup_hists)]["igdb_id"])
    keep_ids.update(unique_hist_ids)

    included = df[df["igdb_id"].isin(keep_ids) | ~has_hist].copy()
    excluded = df[df["igdb_id"].isin(dup_ids)].assign(exclusion_reason="duplicate_price_history")

    return included.reset_index(drop=True), excluded

def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = []

    if "is_hardware" in df.columns:
        mask_hardware = df["is_hardware"].fillna(False).astype(bool)
        if mask_hardware.any():
            excluded.append(df[mask_hardware].assign(exclusion_reason="hardware"))
            df = df[~mask_hardware]

    mask_sparse = df["price_record_count"].fillna(0) < MIN_PRICE_RECORDS
    excluded.append(df[mask_sparse].assign(exclusion_reason=f"fewer_than_{MIN_PRICE_RECORDS}_price_records"))
    df = df[~mask_sparse]

    excluded_df = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    return df.reset_index(drop=True), excluded_df

def main():
    master = pd.read_csv(INPUT_MASTER)
    sales  = pd.read_csv(INPUT_SALES)
    prices = pd.read_csv(INPUT_PRICES)

    sales_cols   = ["igdb_id", "units_sold_global_m", "units_sold_na_m", "match_type", "match_score"]
    sales_subset = sales[sales_cols].drop_duplicates("igdb_id")
    df = master.merge(sales_subset, on="igdb_id", how="left")

    price_features = build_price_features(prices)
    df = df.merge(price_features, on="pc_url", how="left")

    def first_sale_year(hist_json: str) -> int | None:
        try:
            d = ast.literal_eval(hist_json)
            if not d:
                return None
            yr = int(min(d.keys())[:4])
            return yr if yr >= 2005 else None
        except Exception:
            return None

    def corrected_release_year(row) -> int | None:
        igdb_yr = row["release_year"]
        hist_yr = row["_first_sale_year"]
        candidates = []
        if pd.notna(igdb_yr):
            y = int(igdb_yr)
            if y >= 2005:
                candidates.append(y)
        if pd.notna(hist_yr):
            candidates.append(int(hist_yr))
        if candidates:
            return min(candidates)
        # fall back to igdb year even if pre-2005 (avoids losing the row)
        return int(igdb_yr) if pd.notna(igdb_yr) else None

    has_hist = df["price_history_json"].notna()
    df["_first_sale_year"] = None
    df.loc[has_hist, "_first_sale_year"] = df.loc[has_hist, "price_history_json"].apply(first_sale_year)
    df["release_year"] = df.apply(corrected_release_year, axis=1)
    df.drop(columns=["_first_sale_year"], inplace=True)

    variant_info   = df["title"].apply(classify_variant)
    df["is_variant"]   = variant_info.apply(lambda x: x[0])
    df["variant_type"] = variant_info.apply(lambda x: x[1])

    included, excluded = apply_exclusions(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    included.to_csv(OUTPUT_MERGED, index=False)
    print(f"Saved to {OUTPUT_MERGED}")
    if not excluded.empty:
        excluded.to_csv(OUTPUT_EXCLUDED, index=False)
        print(f"Saved to {OUTPUT_EXCLUDED}")

if __name__ == "__main__":
    main()
