import ast
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("data")
INPUT_FILE = DATA_DIR / "merged_dataset.csv"
CPI_FILE   = DATA_DIR / "cpi.csv"

RECENT_CUTOFF = pd.Timestamp("2023-01-01")

EVAL_MODE = "--eval" in sys.argv

_cutoff_year   = 2023
_test_end_year = 2026
for i, arg in enumerate(sys.argv):
    if arg == "--cutoff"   and i + 1 < len(sys.argv):
        _cutoff_year   = int(sys.argv[i + 1])
    if arg == "--test-end" and i + 1 < len(sys.argv):
        _test_end_year = int(sys.argv[i + 1])

PRICE_CUTOFF    = pd.Timestamp(f"{_cutoff_year}-12-31")   if EVAL_MODE else None
TEST_END_CUTOFF = pd.Timestamp(f"{_test_end_year}-12-31") if EVAL_MODE else None
OUTPUT_FILE     = DATA_DIR / f"merged_dataset_eval_{_cutoff_year}.csv" if EVAL_MODE else INPUT_FILE

REFERENCE_YEAR = _cutoff_year if EVAL_MODE else 2025
CURRENT_YEAR   = 2025

MSRP_BY_CONSOLE = {
    "Game Boy Color":   29.99,
    "Game Boy Advance": 29.99,
    "GameCube":         49.99,
    "Nintendo DS":      34.99,
    "Wii":              49.99,
    "Wii U":            59.99,
    "Nintendo 3DS":     39.99,
}

DISCONTINUED_YEAR = {
    "Game Boy Color":   2003,
    "Game Boy Advance": 2008,
    "GameCube":         2007,
    "Nintendo DS":      2014,
    "Wii":              2017,
    "Wii U":            2017,
    "Nintendo 3DS":     2020,
}

LAUNCH_YEAR_BY_CONSOLE = {
    "Game Boy Color":   1998,
    "Game Boy Advance": 2001,
    "GameCube":         2001,
    "Nintendo DS":      2004,
    "Wii":              2006,
    "Wii U":            2012,
    "Nintendo 3DS":     2011,
}

def load_cpi(path: Path) -> dict[int, float]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    year_col = next(c for c in df.columns if c.lower() in ("year", "yr"))
    cpi_col  = next(c for c in df.columns if c.lower() in ("annual", "avg", "value", "cpi"))
    return dict(zip(df[year_col].astype(int), df[cpi_col].astype(float)))

def deflate(nominal_price: float, nominal_year: int, cpi: dict) -> float | None:
    if pd.isna(nominal_price) or CURRENT_YEAR not in cpi:
        return None
    # Forward-fill CPI for years beyond the data (treat as same purchasing power as latest year)
    cpi_year = nominal_year if nominal_year in cpi else max((y for y in cpi if y <= nominal_year), default=None)
    if cpi_year is None:
        return None
    return nominal_price * (cpi[CURRENT_YEAR] / cpi[cpi_year])

def parse_price_history(json_str: str, cutoff: pd.Timestamp | None = None) -> dict[int, float]:
    if not isinstance(json_str, str) or not json_str.strip():
        return {}
    try:
        raw    = ast.literal_eval(json_str)
        annual = {}
        counts = {}
        for date_str, price in raw.items():
            if cutoff is not None and pd.Timestamp(date_str) > cutoff:
                continue
            year = int(date_str[:4])
            annual[year] = annual.get(year, 0.0) + float(price)
            counts[year] = counts.get(year, 0) + 1
        return {y: annual[y] / counts[y] for y in annual}
    except Exception:
        return {}

def parse_price_history_monthly(json_str: str, cutoff: pd.Timestamp | None = None) -> pd.Series:
    if not isinstance(json_str, str) or not json_str.strip():
        return pd.Series(dtype=float)
    try:
        raw  = ast.literal_eval(json_str)
        data = {}
        for d, p in raw.items():
            ts = pd.Timestamp(d)
            if cutoff is not None and ts > cutoff:
                continue
            data[ts] = float(p)
        return pd.Series(data).sort_index()
    except Exception:
        return pd.Series(dtype=float)

def parse_target_price(json_str: str, cutoff: pd.Timestamp, cpi: dict,
                        test_end: pd.Timestamp | None = None) -> float | None:
    if not isinstance(json_str, str) or not json_str.strip():
        return None
    try:
        raw = ast.literal_eval(json_str)
        post_cutoff = {
            pd.Timestamp(d): float(p)
            for d, p in raw.items()
            if pd.Timestamp(d) > cutoff
            and (test_end is None or pd.Timestamp(d) <= test_end)
            and float(p) > 0
        }
        if not post_cutoff:
            return None
        by_year = {}
        counts  = {}
        for ts, price in post_cutoff.items():
            yr = ts.year
            by_year[yr] = by_year.get(yr, 0.0) + price
            counts[yr]  = counts.get(yr, 0) + 1
        real_vals = []
        for yr, total in by_year.items():
            avg_nominal = total / counts[yr]
            real = deflate(avg_nominal, yr, cpi)
            if real is not None:
                real_vals.append(real)
        return round(float(np.mean(real_vals)), 2) if real_vals else None
    except Exception:
        return None

def parse_target_price_dec(json_str: str, years: list[int], cpi: dict) -> dict[int, float | None]:
    result = {y: None for y in years}
    if not isinstance(json_str, str) or not json_str.strip():
        return result
    try:
        raw  = ast.literal_eval(json_str)
        by_ym: dict[tuple[int, int], float] = {}
        for d, p in raw.items():
            ts  = pd.Timestamp(d)
            val = float(p)
            if val > 0:
                by_ym[(ts.year, ts.month)] = val
        for yr in years:
            for mo in (12, 11, 10):
                if (yr, mo) in by_ym:
                    real = deflate(by_ym[(yr, mo)], yr, cpi)
                    result[yr] = round(real, 2) if real is not None else None
                    break
    except Exception:
        pass
    return result

def price_velocity(real_prices_by_year: dict) -> float | None:
    if len(real_prices_by_year) < 3:
        return None
    years  = np.array(sorted(real_prices_by_year.keys()), dtype=float)
    prices = np.array([real_prices_by_year[y] for y in years.astype(int)], dtype=float)
    slope  = np.polyfit(years, prices, 1)[0]
    return round(float(slope), 4)

def _monthly_series(json_str: str, cutoff: pd.Timestamp | None,
                    cpi: dict, months: int) -> list[tuple]:
    try:
        raw = ast.literal_eval(json_str) if isinstance(json_str, str) else {}
        if not raw:
            return []
        end   = cutoff if cutoff is not None else pd.Timestamp.now().normalize()
        start = end - pd.DateOffset(months=months)
        result = []
        for d, p in raw.items():
            ts = pd.Timestamp(d)
            pv = float(p)
            if start <= ts <= end and pv > 0:
                real = deflate(pv, ts.year, cpi)
                if real is not None:
                    result.append((ts, real))
        return sorted(result)
    except Exception:
        return []

def velocity_window(json_str: str, cutoff: pd.Timestamp | None,
                    cpi: dict, months: int) -> float | None:
    series = _monthly_series(json_str, cutoff, cpi, months)
    if len(series) < 3:
        return None
    t0 = series[0][0]
    xs = np.array([(t - t0).days for t, _ in series], dtype=float)
    ys = np.array([p for _, p in series], dtype=float)
    if xs[-1] == 0:
        return None
    slope = np.polyfit(xs, ys, 1)[0] * 365.25  # annualised $/yr
    return round(float(slope), 4)

def volatility_window(json_str: str, cutoff: pd.Timestamp | None,
                      cpi: dict, months: int) -> float | None:
    series = _monthly_series(json_str, cutoff, cpi, months)
    if len(series) < 3:
        return None
    prices  = [p for _, p in series]
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    return round(float(np.std(changes)), 4) if len(changes) >= 2 else (round(changes[0], 4) if changes else None)

def price_volatility(real_prices_by_year: dict) -> float | None:
    if len(real_prices_by_year) < 2:
        return None
    return round(float(np.std(list(real_prices_by_year.values()))), 4)

def price_at_year_offset(real_prices: dict, release_year: int, offset: int) -> float | None:
    target = int(release_year) + offset
    return real_prices.get(target)

def price_at_month_offset(json_str: str, cutoff: pd.Timestamp | None,
                           cpi: dict, release_year: int, months: int) -> float | None:
    """Real price closest to release_date + months (within ±3 months)."""
    if pd.isna(release_year):
        return None
    target = pd.Timestamp(year=int(release_year), month=1, day=1) + pd.DateOffset(months=months)
    if cutoff is not None and target > cutoff:
        return None
    try:
        raw = ast.literal_eval(json_str) if isinstance(json_str, str) else {}
        best_val, best_diff = None, float("inf")
        for d, p in raw.items():
            ts = pd.Timestamp(d)
            if cutoff is not None and ts > cutoff:
                continue
            pv = float(p)
            if pv <= 0:
                continue
            diff = abs((ts - target).days)
            if diff <= 91 and diff < best_diff:
                real = deflate(pv, ts.year, cpi)
                if real is not None:
                    best_val, best_diff = real, diff
        return round(best_val, 2) if best_val is not None else None
    except Exception:
        return None

def price_change_window(json_str: str, cutoff: pd.Timestamp | None,
                         cpi: dict, months: int) -> float | None:
    """Absolute real-price change over the last N months."""
    series = _monthly_series(json_str, cutoff, cpi, months)
    if len(series) < 2:
        return None
    return round(series[-1][1] - series[0][1], 4)

def price_pct_change_window(json_str: str, cutoff: pd.Timestamp | None,
                             cpi: dict, months: int) -> float | None:
    """Fractional real-price change over the last N months."""
    series = _monthly_series(json_str, cutoff, cpi, months)
    if len(series) < 2:
        return None
    start = series[0][1]
    if start <= 0:
        return None
    return round((series[-1][1] - start) / start, 4)

def price_accel(json_str: str, cutoff: pd.Timestamp | None, cpi: dict) -> float | None:
    """Change in velocity: recent-3m velocity minus prior-3m velocity (annualised $/yr)."""
    series = _monthly_series(json_str, cutoff, cpi, 6)
    if len(series) < 4:
        return None
    mid = len(series) // 2

    def _vel(s: list) -> float | None:
        if len(s) < 2:
            return None
        t0 = s[0][0]
        xs = np.array([(t - t0).days for t, _ in s], dtype=float)
        ys = np.array([p for _, p in s], dtype=float)
        return np.polyfit(xs, ys, 1)[0] * 365.25 if xs[-1] > 0 else None

    v_rec = _vel(series[mid:])
    v_old = _vel(series[:mid])
    if v_rec is None or v_old is None:
        return None
    return round(v_rec - v_old, 4)

def price_vs_range(json_str: str, cutoff: pd.Timestamp | None,
                   cpi: dict, months: int = 12) -> tuple[float | None, float | None]:
    """(current / 52w-high, current / 52w-low) ratios."""
    series = _monthly_series(json_str, cutoff, cpi, months)
    if len(series) < 2:
        return None, None
    prices  = [p for _, p in series]
    current = prices[-1]
    high    = max(prices)
    low     = min(prices)
    vs_high = round(current / high, 4) if high > 0 else None
    vs_low  = round(current / low,  4) if low  > 0 else None
    return vs_high, vs_low

def engineer_features(df: pd.DataFrame, cpi: dict,
                      price_cutoff: pd.Timestamp | None = None,
                      reference_year: int = 2024,
                      test_end_cutoff: pd.Timestamp | None = None) -> pd.DataFrame:

    df["msrp_nominal"]    = df["console"].map(MSRP_BY_CONSOLE)
    df["msrp_launch_year"] = df["console"].map(LAUNCH_YEAR_BY_CONSOLE)
    df["real_msrp"] = df.apply(
        lambda r: deflate(r["msrp_nominal"], r["msrp_launch_year"], cpi), axis=1
    )

    real_prices_all = {}
    for _, row in df.iterrows():
        nominal_by_year = parse_price_history(row.get("price_history_json", ""), cutoff=price_cutoff)
        real_by_year = {
            y: deflate(p, y, cpi)
            for y, p in nominal_by_year.items()
            if deflate(p, y, cpi) is not None
        }
        real_prices_all[row["igdb_id"]] = real_by_year

    all_year_prices = {}
    for prices in real_prices_all.values():
        for year, price in prices.items():
            all_year_prices.setdefault(year, []).append(price)
    market_index = {year: float(np.median(vals)) for year, vals in all_year_prices.items()}

    rows_idx = df.index.tolist()
    WINDOWS = [("1m", 1), ("3m", 3), ("6m", 6), ("24m", 24), ("60m", 60)]

    feat_price = {
        "real_price_at_12m":       [],
        "real_price_at_60m":       [],
        "price_velocity":          [],
        "price_volatility":        [],
        **{f"price_velocity_{w}":  [] for w, _ in WINDOWS},
        **{f"price_volatility_{w}":[] for w, _ in WINDOWS},
        "price_change_1m":         [],
        "price_change_3m":         [],
        "price_change_6m":         [],
        "price_change_12m":        [],
        "price_pct_change_1m":     [],
        "price_pct_change_3m":     [],
        "price_pct_change_6m":     [],
        "price_pct_change_12m":    [],
        "price_vs_52w_high":       [],
        "price_vs_52w_low":        [],
        "price_acceleration":      [],
        "real_msrp_ratio":         [],
        "price_vs_market_mean":    [],
        "market_liquidity_score":  [],
    }

    for idx in rows_idx:
        row          = df.loc[idx]
        igdb_id      = row["igdb_id"]
        release_year = row.get("release_year")
        real_prices  = real_prices_all.get(igdb_id, {})
        hist_json    = row.get("price_history_json", "{}")

        at12m = price_at_month_offset(hist_json, price_cutoff, cpi, release_year, 12)
        at60m = price_at_month_offset(hist_json, price_cutoff, cpi, release_year, 60)
        # Full-history velocity/volatility using monthly data (240m = 20-year cap)
        vel = velocity_window(hist_json, price_cutoff, cpi, 240)
        vol = volatility_window(hist_json, price_cutoff, cpi, 240)
        rmsrp = row.get("real_msrp")

        ratios = []
        for year, price in real_prices.items():
            mi = market_index.get(year)
            if mi and mi > 0:
                ratios.append(price / mi)
        pvm = float(np.mean(ratios)) if ratios else None

        record_count = row.get("price_record_count", 0) or 0

        h52_high, h52_low = price_vs_range(hist_json, price_cutoff, cpi, 12)

        feat_price["real_price_at_12m"].append(at12m)
        feat_price["real_price_at_60m"].append(at60m)
        feat_price["price_velocity"].append(vel)
        feat_price["price_volatility"].append(vol)
        for w, m in WINDOWS:
            feat_price[f"price_velocity_{w}"].append(velocity_window(hist_json, price_cutoff, cpi, m))
            feat_price[f"price_volatility_{w}"].append(volatility_window(hist_json, price_cutoff, cpi, m))
        feat_price["price_change_1m"].append(price_change_window(hist_json, price_cutoff, cpi, 1))
        feat_price["price_change_3m"].append(price_change_window(hist_json, price_cutoff, cpi, 3))
        feat_price["price_change_6m"].append(price_change_window(hist_json, price_cutoff, cpi, 6))
        feat_price["price_change_12m"].append(price_change_window(hist_json, price_cutoff, cpi, 12))
        feat_price["price_pct_change_1m"].append(price_pct_change_window(hist_json, price_cutoff, cpi, 1))
        feat_price["price_pct_change_3m"].append(price_pct_change_window(hist_json, price_cutoff, cpi, 3))
        feat_price["price_pct_change_6m"].append(price_pct_change_window(hist_json, price_cutoff, cpi, 6))
        feat_price["price_pct_change_12m"].append(price_pct_change_window(hist_json, price_cutoff, cpi, 12))
        feat_price["price_vs_52w_high"].append(h52_high)
        feat_price["price_vs_52w_low"].append(h52_low)
        feat_price["price_acceleration"].append(price_accel(hist_json, price_cutoff, cpi))
        feat_price["real_msrp_ratio"].append(
            round(at60m / rmsrp, 4) if at60m and rmsrp and rmsrp > 0 else None
        )
        feat_price["price_vs_market_mean"].append(round(pvm, 4) if pvm is not None else None)
        feat_price["market_liquidity_score"].append(record_count)

    for col, vals in feat_price.items():
        df[col] = vals

    df["months_since_release"] = df["release_year"].apply(
        lambda y: (reference_year - int(y)) * 12 if pd.notna(y) else None
    )
    df["months_since_discontinued"] = df["console"].map(DISCONTINUED_YEAR).apply(
        lambda disc_yr: (reference_year - int(disc_yr)) * 12 if pd.notna(disc_yr) else None
    )

    df["units_sold_best"] = df["units_sold_na_m"].combine_first(df["units_sold_global_m"])
    df["units_sold_log"]  = np.log1p(df["units_sold_best"] * 1e6)

    df["days_since_last_franchise_release"] = None
    df["days_until_next_franchise_release"] = None
    df["franchise_entry_count"]             = None
    df["franchise_releases_in_2yr_window"]  = None

    fran_rows = []
    for _, row in df.iterrows():
        ids_str = str(row.get("franchise_igdb_ids", ""))
        if ids_str and ids_str not in ("nan", ""):
            for fid in ids_str.split("|"):
                fran_rows.append({"igdb_id": row["igdb_id"], "franchise_id": fid.strip(),
                                   "release_year": row["release_year"]})

    if fran_rows:
        fdf = pd.DataFrame(fran_rows)
        fdf["release_year"] = pd.to_numeric(fdf["release_year"], errors="coerce")
        fdf = fdf.dropna(subset=["release_year"])
        fdf = fdf.sort_values(["franchise_id", "release_year"])

        entry_counts = fdf.groupby("franchise_id")["igdb_id"].transform("count")
        fdf["franchise_entry_count"] = entry_counts
        fdf["prev_year"]             = fdf.groupby("franchise_id")["release_year"].shift(1)
        fdf["next_year"]             = fdf.groupby("franchise_id")["release_year"].shift(-1)
        fdf["days_since_last"]       = (fdf["release_year"] - fdf["prev_year"]) * 365
        fdf["days_until_next"]       = (fdf["next_year"] - fdf["release_year"]) * 365

        def count_in_window(sub):
            results = []
            yrs = sub["release_year"].values
            for y in yrs:
                cnt = np.sum((yrs >= y - 2) & (yrs <= y + 2)) - 1
                results.append(cnt)
            sub = sub.copy()
            sub["in_window"] = results
            return sub

        fdf = fdf.groupby("franchise_id", group_keys=False).apply(count_in_window)

        fdf_agg = fdf.groupby("igdb_id").agg(
            franchise_entry_count=("franchise_entry_count", "max"),
            days_since_last_franchise_release=("days_since_last", "min"),
            days_until_next_franchise_release=("days_until_next", "min"),
            franchise_releases_in_2yr_window=("in_window", "max"),
        ).reset_index()

        df = df.merge(fdf_agg, on="igdb_id", how="left", suffixes=("_old", ""))
        for col in ["franchise_entry_count", "days_since_last_franchise_release",
                    "days_until_next_franchise_release", "franchise_releases_in_2yr_window"]:
            if col + "_old" in df.columns:
                df.drop(columns=[col + "_old"], inplace=True)

    def _latest_monthly_real(json_str: str) -> float | None:
        series = parse_price_history_monthly(json_str, cutoff=price_cutoff)
        if series.empty:
            return None
        last_ts = series.index.max()
        return round(deflate(float(series[last_ts]), last_ts.year, cpi) or 0, 2) or None

    df["current_real_price"] = df["price_history_json"].apply(_latest_monthly_real)
    df.drop(columns=["genre_median_price"], errors="ignore", inplace=True)
    genre_medians = df.groupby("genre")["current_real_price"].median().rename("genre_median_price")
    df = df.merge(genre_medians.reset_index(), on="genre", how="left")
    df["price_vs_genre"] = df.apply(
        lambda r: round(r["current_real_price"] / r["genre_median_price"], 4)
        if pd.notna(r.get("current_real_price")) and r.get("genre_median_price", 0) > 0
        else None,
        axis=1,
    )

    df["same_genre_same_platform_top_units_in_window"]     = None
    df["same_genre_same_platform_release_count_in_window"] = None
    df["same_genre_cross_platform_top_units_in_window"]    = None
    df["units_sold_vs_top_competitor"]                     = None

    df["release_year_int"] = pd.to_numeric(df["release_year"], errors="coerce")
    for idx in df.index:
        row = df.loc[idx]
        if pd.isna(row["release_year_int"]) or pd.isna(row["genre"]):
            continue
        yr  = row["release_year_int"]
        gen = row["genre"]
        con = row["console"]

        mask_sp  = (
            (df["genre"] == gen) &
            (df["console"] == con) &
            (df["release_year_int"].between(yr - 1, yr + 1)) &
            (df["igdb_id"] != row["igdb_id"])
        )
        sp_group = df[mask_sp]
        top_sp   = sp_group["units_sold_best"].max() if not sp_group.empty else None
        count_sp = len(sp_group)

        mask_cp = (
            (df["genre"] == gen) &
            (df["console"] != con) &
            (df["release_year_int"].between(yr - 1, yr + 1))
        )
        top_cp    = df[mask_cp]["units_sold_best"].max() if not df[mask_cp].empty else None
        own_units = row["units_sold_best"]
        vs_top    = (own_units / top_sp) if (top_sp and top_sp > 0 and pd.notna(own_units)) else None

        df.at[idx, "same_genre_same_platform_top_units_in_window"]     = top_sp
        df.at[idx, "same_genre_same_platform_release_count_in_window"] = count_sp
        df.at[idx, "same_genre_cross_platform_top_units_in_window"]    = top_cp
        df.at[idx, "units_sold_vs_top_competitor"]                     = round(vs_top, 4) if vs_top else None

    df["publisher"] = df["publisher"].fillna("Unknown")

    if price_cutoff is not None:
        forecast_end = test_end_cutoff if test_end_cutoff is not None else pd.Timestamp(f"{price_cutoff.year + 2}-12-01")
        forecast_months = pd.date_range(
            f"{price_cutoff.year + 1}-01-01", end=forecast_end, freq="MS"
        )

        def extract_monthly_targets(json_str: str) -> str:
            targets: dict[str, float] = {}
            if not isinstance(json_str, str) or not json_str.strip():
                return "{}"
            try:
                raw = ast.literal_eval(json_str)
                for ts in forecast_months:
                    key = ts.strftime("%Y-%m-01")
                    val = raw.get(key)
                    if val is not None and float(val) > 0:
                        real = deflate(float(val), ts.year, cpi)
                        if real is not None:
                            targets[ts.strftime("%Y-%m")] = round(real, 2)
            except Exception:
                pass
            return str(targets)

        df["target_months_json"] = df["price_history_json"].apply(extract_monthly_targets)

        def first_monthly_target(s: str) -> float | None:
            try:
                d = ast.literal_eval(s)
                return list(d.values())[0] if d else None
            except Exception:
                return None

        df["target_real_price"] = df["target_months_json"].apply(first_monthly_target)
    else:
        df["target_real_price"] = df["current_real_price"]

    df.drop(columns=[
        "msrp_launch_year", "release_year_int", "units_sold_best",
    ], errors="ignore", inplace=True)

    return df

def main():
    df = pd.read_csv(INPUT_FILE)

    if not CPI_FILE.exists():
        raise FileNotFoundError(
            f"{CPI_FILE} not found. Save BLS CPI-U annual averages as data/cpi.csv "
            "with columns 'Year' and 'Annual'."
        )

    cpi = load_cpi(CPI_FILE)

    df = engineer_features(df, cpi, price_cutoff=PRICE_CUTOFF, reference_year=REFERENCE_YEAR,
                           test_end_cutoff=TEST_END_CUTOFF)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
