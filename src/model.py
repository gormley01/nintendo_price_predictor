import ast
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore", category=UserWarning)

import sys as _sys

_cutoff_year = 2023
for i, a in enumerate(_sys.argv):
    if a == "--cutoff" and i + 1 < len(_sys.argv):
        _cutoff_year = int(_sys.argv[i + 1])

DATA_DIR       = Path("data")
MODELS_DIR     = DATA_DIR / "models"
CPI_FILE       = DATA_DIR / "cpi.csv"
EVAL_INPUT     = DATA_DIR / f"merged_dataset_eval_{_cutoff_year}.csv"
PROD_INPUT     = DATA_DIR / "merged_dataset.csv"
OUTPUT_PREDS   = DATA_DIR / "predictions.csv"
OUTPUT_SHAP    = DATA_DIR / "shap_values.csv"
OUTPUT_METRICS = DATA_DIR / "metrics.json"

RANDOM_STATE         = 42
CONFIDENCE_THRESHOLD = 0.50

PROD_FORECAST_START  = pd.Timestamp("2025-01-01")
PROD_FORECAST_END    = pd.Timestamp("2030-12-01")
PROD_FORECAST_MONTHS = len(pd.date_range(PROD_FORECAST_START, PROD_FORECAST_END, freq="MS"))

HIST_LOOKBACK_MONTHS = 36

NUMERIC_FEATURES = [
    "prev_real_price",
    "current_real_price",
    "months_since_release",
    "months_since_discontinued",
    "real_price_at_12m",
    "real_price_at_60m",
    "price_velocity",
    "price_volatility",
    "price_velocity_1m",
    "price_volatility_1m",
    "price_velocity_3m",
    "price_volatility_3m",
    "price_velocity_6m",
    "price_volatility_6m",
    "price_velocity_24m",
    "price_volatility_24m",
    "price_velocity_60m",
    "price_volatility_60m",
    "price_change_1m",
    "price_change_3m",
    "price_change_6m",
    "price_change_12m",
    "price_pct_change_1m",
    "price_pct_change_3m",
    "price_pct_change_6m",
    "price_pct_change_12m",
    "price_vs_52w_high",
    "price_vs_52w_low",
    "price_acceleration",
    "real_msrp_ratio",
    "price_vs_market_mean",
    "market_liquidity_score",
    "units_sold_log",
    "franchise_entry_count",
    "days_since_last_franchise_release",
    "days_until_next_franchise_release",
    "franchise_releases_in_2yr_window",
    "genre_median_price",
    "price_vs_genre",
    "same_genre_same_platform_top_units_in_window",
    "same_genre_same_platform_release_count_in_window",
    "same_genre_cross_platform_top_units_in_window",
    "units_sold_vs_top_competitor",
    "igdb_critic_score",
    "igdb_user_score",
    "critic_user_score_delta",
    "rerelease_critic_score",
    "rerelease_user_score",
    "rerelease_critic_user_delta",
    "rerelease_units_sold",
    "months_ahead",
]

CATEGORICAL_FEATURES = [
    "console",
    "genre",
    "esrb",
    "canonical_condition",
    "rerelease_type",
]

BINARY_FEATURES = [
    "rerelease_exists",
]

TARGET = "target_real_price"

def load_cpi_for_model() -> dict[int, float]:
    if not CPI_FILE.exists():
        return {}
    df = pd.read_csv(CPI_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    year_col = next((c for c in df.columns if c in ("year", "yr")), None)
    cpi_col  = next((c for c in df.columns if c in ("annual", "avg", "value", "cpi")), None)
    if not year_col or not cpi_col:
        return {}
    return dict(zip(df[year_col].astype(int), df[cpi_col].astype(float)))

def deflate_price(price: float, year: int, cpi: dict, ref_year: int = 2025) -> float:
    if not cpi or ref_year not in cpi:
        return float(price)
    # Forward-fill CPI for years beyond the data
    cpi_year = year if year in cpi else max((y for y in cpi if y <= year), default=None)
    if cpi_year is None or cpi[cpi_year] == 0:
        return float(price)
    return float(price) * (cpi[ref_year] / cpi[cpi_year])

def confidence_pct(interval_width: np.ndarray, median: np.ndarray,
                   months_ahead: np.ndarray | None = None) -> np.ndarray:
    ratio = interval_width / np.maximum(median, 0.01)
    return np.clip(np.round(100.0 * (1.0 - ratio / 2.0), 1), 0.0, 100.0)

def mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def directional_accuracy(y_true, y_pred, baseline):
    mask          = baseline.notna() & (baseline > 0)
    actual_dir    = (y_true[mask] > baseline[mask]).astype(int)
    predicted_dir = (y_pred[mask] > baseline[mask]).astype(int)
    return float((actual_dir == predicted_dir).mean())

def interval_coverage(y_true, lower, upper):
    return float(((y_true >= lower) & (y_true <= upper)).mean())

def catalog_coverage(interval_widths, medians, threshold=CONFIDENCE_THRESHOLD):
    ratios = interval_widths / medians.clip(lower=0.01)
    return float((ratios <= threshold).mean())

def train_lgb_quantile(X_train, y_train, X_val, y_val, alpha: float,
                       weights=None) -> lgb.Booster:
    params = {
        "objective":        "quantile",
        "alpha":            alpha,
        "metric":           "quantile",
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)
    return lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

def encode_publishers(df_train: pd.DataFrame, df_val: pd.DataFrame):
    enc = TargetEncoder(target_type="continuous", smooth="auto", random_state=42)
    enc.fit(df_train[["publisher"]], df_train[TARGET])
    df_train["publisher_encoded"] = enc.transform(df_train[["publisher"]]).ravel()
    df_val["publisher_encoded"]   = enc.transform(df_val[["publisher"]]).ravel()
    return enc

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + ["publisher_encoded"]
    available    = [c for c in all_features if c in df.columns]
    X            = df[available].copy()
    for col in NUMERIC_FEATURES + ["publisher_encoded"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    for col in BINARY_FEATURES:
        if col in X.columns:
            X[col] = X[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    return X

def split_by_game(df: pd.DataFrame, test_size: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_ids = df["igdb_id"].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=RANDOM_STATE)
    return df[df["igdb_id"].isin(train_ids)].copy(), df[df["igdb_id"].isin(val_ids)].copy()

def expand_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "target_months_json" not in df.columns:
        return pd.DataFrame()
    records = []
    for _, row in df.iterrows():
        try:
            targets: dict = ast.literal_eval(str(row["target_months_json"]))
        except Exception:
            continue
        if not targets:
            continue
        sorted_targets = sorted(targets.items())
        current_price  = row.get("current_real_price")
        for month_i, (month_str, real_price) in enumerate(sorted_targets, start=1):
            prev_price_val = current_price if month_i == 1 else sorted_targets[month_i - 2][1]
            if prev_price_val is None or pd.isna(prev_price_val):
                continue
            prev_f = float(prev_price_val)
            real_f = float(real_price)
            if prev_f <= 0 or real_f <= 0:
                continue
            rec = row.to_dict()
            rec["months_ahead"]       = month_i
            rec["prediction_date"]    = month_str + "-01"
            rec[TARGET]               = round(float(np.log(real_f / prev_f)), 6)
            rec["prev_real_price"]    = round(prev_f, 2)
            rec["actual_real_price"]  = round(float(real_price), 2)
            records.append(rec)
    return pd.DataFrame(records)

def expand_hist_rows(df: pd.DataFrame, cpi: dict,
                     reference_date: pd.Timestamp,
                     lookback_months: int) -> pd.DataFrame:
    cutoff_lo = reference_date - pd.DateOffset(months=lookback_months)
    records   = []
    for _, row in df.iterrows():
        try:
            raw: dict = ast.literal_eval(str(row.get("price_history_json", "") or "{}"))
        except Exception:
            continue
        # Build full real-price series for prev_price lookups
        full_real: dict[pd.Timestamp, float] = {}
        for d, p in raw.items():
            try:
                ts_p = pd.Timestamp(d)
                pv   = float(p)
                if pv > 0:
                    full_real[ts_p] = deflate_price(pv, ts_p.year, cpi)
            except Exception:
                continue
        sorted_ts = sorted(full_real.keys())

        for date_str, price in raw.items():
            try:
                ts        = pd.Timestamp(date_str)
                price_val = float(price)
            except Exception:
                continue
            if ts < cutoff_lo or ts >= reference_date or price_val <= 0:
                continue
            real   = deflate_price(price_val, ts.year, cpi)
            months = (ts.year - reference_date.year) * 12 + (ts.month - reference_date.month)

            # Previous month real price (look back up to 45 days)
            prev_real = full_real.get(ts - pd.DateOffset(months=1))
            if prev_real is None:
                prior = [d for d in sorted_ts if d < ts]
                if prior and (ts - max(prior)).days <= 45:
                    prev_real = full_real[max(prior)]

            if prev_real is None or prev_real <= 0 or real is None or real <= 0:
                continue  # need prev price to compute log-return
            rec = row.to_dict()
            rec["months_ahead"]    = months
            rec["prediction_date"] = ts.strftime("%Y-%m-01")
            rec[TARGET]            = round(float(np.log(real / prev_real)), 6)
            rec["prev_real_price"] = round(prev_real, 2)
            records.append(rec)
    return pd.DataFrame(records)

def _last_price_date(json_str) -> pd.Timestamp | None:
    try:
        d = ast.literal_eval(str(json_str))
        if not d:
            return None
        return max(pd.Timestamp(k) for k in d.keys())
    except Exception:
        return None

def expand_inference_rows(df: pd.DataFrame, n_months: int,
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    days_per_month = 30.44
    records = []
    for _, row in df.iterrows():
        base = row.to_dict()
        # start from month after last known price, fall back to global start_date
        last_dt = _last_price_date(base.get("price_history_json"))
        if last_dt is not None:
            game_start = (last_dt + pd.DateOffset(months=1)).replace(day=1)
            game_start = max(game_start, start_date)
        else:
            game_start = start_date

        if end_date is not None:
            game_dates = pd.date_range(game_start, end=end_date, freq="MS")
        else:
            game_dates = pd.date_range(game_start, periods=n_months, freq="MS")

        for months_ahead, pred_date in enumerate(game_dates, start=1):
            rec = base.copy()
            rec["months_ahead"]    = months_ahead
            rec["prediction_date"] = pred_date.strftime("%Y-%m-01")
            day_delta = months_ahead * days_per_month
            if pd.notna(base.get("months_since_release")):
                rec["months_since_release"] = base["months_since_release"] + months_ahead
            if pd.notna(base.get("months_since_discontinued")):
                rec["months_since_discontinued"] = base["months_since_discontinued"] + months_ahead
            if pd.notna(base.get("days_since_last_franchise_release")):
                rec["days_since_last_franchise_release"] = base["days_since_last_franchise_release"] + day_delta
            if pd.notna(base.get("days_until_next_franchise_release")):
                rec["days_until_next_franchise_release"] = max(0.0, base["days_until_next_franchise_release"] - day_delta)
            records.append(rec)
    return pd.DataFrame(records)

def predict_quantiles(
    models: dict, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_mid = models["median"].predict(X)
    q10 = np.minimum(models["lower"].predict(X), pred_mid)
    q90 = np.maximum(models["upper"].predict(X), pred_mid)
    return pred_mid, q10, q90

def build_prediction_df(base_cols: pd.DataFrame,
                        pred_mid, q10, q90,
                        actual_price=None,
                        last_known_price=None) -> pd.DataFrame:
    widths = q90 - q10
    out = base_cols.copy()
    out["lower_bound"]       = np.round(q10,      2)
    out["prediction"]        = np.round(pred_mid,  2)
    out["upper_bound"]       = np.round(q90,       2)
    out["interval_width"]    = np.round(widths,    2)
    ma = out["months_ahead"].values if "months_ahead" in out.columns else None
    out["confidence_pct"]    = confidence_pct(widths, pred_mid, ma)
    if actual_price is not None:
        out["actual_price"] = np.round(actual_price, 2)
    if last_known_price is not None:
        out["last_known_price"] = np.round(pd.to_numeric(pd.Series(last_known_price), errors="coerce").values, 2)
    return out

def eval_horizon_metrics(y_true: np.ndarray, pred_mid: np.ndarray, q10: np.ndarray,
                          q90: np.ndarray, baseline: np.ndarray, label: str) -> dict:
    mask = pd.notna(y_true) & (y_true > 0) & pd.notna(pred_mid) & np.isfinite(pred_mid)
    if mask.sum() < 10:
        return {}
    yt, ym = y_true[mask], pred_mid[mask]
    yw     = q90[mask] - q10[mask]
    yb     = pd.Series(baseline[mask])

    return {
        "n":                      int(mask.sum()),
        "rmse":                   round(float(root_mean_squared_error(yt, ym)), 2),
        "mape_pct":               round(mape(yt, ym), 2),
        "r2":                     round(float(r2_score(yt, ym)), 3),
        "directional_accuracy":   round(directional_accuracy(yt, ym, yb), 3),
        "interval_coverage_80":   round(interval_coverage(yt, q10[mask], q90[mask]), 3),
        "catalog_coverage_50pct": round(catalog_coverage(yw, pd.Series(ym)), 3),
        "mean_confidence_pct":    round(float(confidence_pct(yw, ym).mean()), 1),
    }

def build_training_data(df_base: pd.DataFrame,
                        cpi: dict,
                        reference_date: pd.Timestamp,
                        extra_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    df_hist = expand_hist_rows(df_base, cpi, reference_date, HIST_LOOKBACK_MONTHS)
    parts   = [df_hist]
    if extra_rows is not None and not extra_rows.empty:
        parts.append(extra_rows.dropna(subset=[TARGET]))
    return pd.concat(parts, ignore_index=True).dropna(subset=[TARGET])

def train_phase(df_train_all: pd.DataFrame, phase_label: str,
                sample_weights: dict | None = None) -> tuple[dict, object, list]:
    train_df, val_df = split_by_game(df_train_all)

    enc = encode_publishers(train_df, val_df)

    X_train = prepare_features(train_df)
    X_val   = prepare_features(val_df)
    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values

    row_weights = None
    if sample_weights:
        row_weights = train_df["igdb_id"].apply(
            lambda gid: sample_weights.get(gid, 1.0)
        ).values

    feature_names = list(X_train.columns)

    models = {}
    for alpha, label in [(0.10, "lower"), (0.50, "median"), (0.90, "upper")]:
        models[label] = train_lgb_quantile(X_train, y_train, X_val, y_val, alpha,
                                           weights=row_weights)

    return models, enc, feature_names

def kfold_error_weights(df_eval: pd.DataFrame, cpi: dict,
                         eval_start: pd.Timestamp, n_folds: int = 10) -> dict:
    """10-fold CV on eval dataset. Returns per-game weight: higher error → higher weight."""
    unique_ids = df_eval["igdb_id"].dropna().unique()
    rng        = np.random.default_rng(RANDOM_STATE)
    shuffled   = rng.permutation(unique_ids)
    folds      = np.array_split(shuffled, n_folds)

    eval_rows_all = expand_eval_rows(df_eval).dropna(subset=[TARGET])
    actual_map    = (
        eval_rows_all.drop_duplicates(["igdb_id", "months_ahead"])
                     .set_index(["igdb_id", "months_ahead"])[TARGET]
        if not eval_rows_all.empty else pd.Series(dtype=float)
    )

    eval_end     = eval_start + pd.DateOffset(months=23)
    game_errors: dict = {}

    for fold_i, val_ids in enumerate(folds):
        val_set  = set(val_ids.tolist())
        train_ids = set(unique_ids.tolist()) - val_set

        df_tr    = df_eval[df_eval["igdb_id"].isin(train_ids)]
        er_tr    = eval_rows_all[eval_rows_all["igdb_id"].isin(train_ids)]
        df_train = build_training_data(df_tr, cpi, eval_start, extra_rows=er_tr)
        if len(df_train) < 100:
            continue

        print(f"  Fold {fold_i + 1}/{n_folds} — training on {len(train_ids)} games…")
        models_f, enc_f, _ = train_phase(df_train, f"fold_{fold_i}")

        df_val = df_eval[df_eval["igdb_id"].isin(val_set)]
        if df_val.empty:
            continue

        preds = sequential_inference(df_val, models_f, enc_f, eval_start, eval_end,
                                     per_game_start=False)

        for igdb_id in val_set:
            gp = preds[preds["igdb_id"] == igdb_id]
            apes = []
            for _, row in gp.iterrows():
                actual = actual_map.get((igdb_id, row["months_ahead"]))
                if actual is not None and pd.notna(actual) and float(actual) > 0:
                    apes.append(abs(row["prediction"] - float(actual)) / float(actual) * 100)
            if apes:
                game_errors[igdb_id] = float(np.mean(apes))

    if not game_errors:
        return {}

    median_err = float(np.median(list(game_errors.values())))
    weights    = {
        gid: float(np.clip(err / max(median_err, 1.0), 0.5, 3.0))
        for gid, err in game_errors.items()
    }
    print(f"  Weights computed for {len(weights)} games — median MAPE {median_err:.1f}%")
    return weights

def compute_external_projections(
    df: pd.DataFrame,
    cpi: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Fit linear trends to historical market-wide and per-genre real-price medians,
    then project forward month-by-month over"""
    all_year_prices: dict[int, list[float]] = {}
    genre_year_prices: dict[str, dict[int, list[float]]] = {}

    for _, row in df.iterrows():
        hist_json = str(row.get("price_history_json", "") or "")
        genre     = str(row.get("genre", "Unknown") or "Unknown")
        try:
            raw = ast.literal_eval(hist_json) if hist_json.strip() else {}
        except Exception:
            raw = {}
        for d, p in raw.items():
            try:
                ts   = pd.Timestamp(d)
                pv   = float(p)
                if pv <= 0:
                    continue
                real = deflate_price(pv, ts.year, cpi)
                yr   = ts.year
                all_year_prices.setdefault(yr, []).append(real)
                genre_year_prices.setdefault(genre, {}).setdefault(yr, []).append(real)
            except Exception:
                continue

    months = pd.date_range(start, end, freq="MS")

    def _project(year_price_map: dict) -> dict:
        valid = {yr: vals for yr, vals in year_price_map.items() if len(vals) >= 3}
        if len(valid) < 3:
            return {}
        yrs  = np.array(sorted(valid.keys()), dtype=float)
        meds = np.array([float(np.median(valid[int(y)])) for y in yrs], dtype=float)
        slope, intercept = np.polyfit(yrs, meds, 1)
        return {
            m: max(float(slope * (m.year + (m.month - 0.5) / 12) + intercept), 1.0)
            for m in months
        }

    return {
        "market_index":  _project(all_year_prices),
        "genre_medians": {g: _project(gm) for g, gm in genre_year_prices.items()},
    }

def _buf_velocity(buf: list) -> float | None:
    if len(buf) < 2:
        return None
    x = np.arange(len(buf), dtype=float)
    y = np.array(buf, dtype=float)
    slope = np.polyfit(x, y, 1)[0]  # monthly slope
    return round(float(slope * 12), 4)  # annualized

def _buf_volatility(buf: list) -> float | None:
    if len(buf) < 2:
        return None
    return round(float(np.std(buf)), 4)

def sequential_inference(
    df_base: pd.DataFrame, models: dict, enc,
    start_date: pd.Timestamp, end_date: pd.Timestamp,
    per_game_start: bool = True,
    external_proj: dict | None = None,
) -> pd.DataFrame:
    """Month-by-month inference; each month's prediction feeds as prev_real_price for the next"""
    days_per_month = 30.44

    game_starts: dict = {}
    for _, row in df_base.iterrows():
        gid = row["igdb_id"]
        if per_game_start:
            last_dt = _last_price_date(row.get("price_history_json"))
            if last_dt is not None:
                gs = (last_dt + pd.DateOffset(months=1)).replace(day=1)
                game_starts[gid] = max(gs, start_date)
            else:
                game_starts[gid] = start_date
        else:
            game_starts[gid] = start_date

    tv_cols  = ["months_since_release", "months_since_discontinued",
                "days_since_last_franchise_release", "days_until_next_franchise_release"]
    deduped  = df_base.drop_duplicates("igdb_id").set_index("igdb_id")
    base_tv  = {col: deduped[col].to_dict() for col in tv_cols if col in deduped.columns}

    prev_price: dict = deduped["current_real_price"].to_dict() if "current_real_price" in deduped.columns else {}
    months_counter: dict = {}

    market_proj = (external_proj or {}).get("market_index", {})
    genre_proj  = (external_proj or {}).get("genre_medians", {})

    # Rolling buffer of predicted prices per game, seeded with the initial price.
    # Used to dynamically update velocity/volatility so predictions show trends.
    predicted_buffer: dict = {
        gid: [float(p)] for gid, p in prev_price.items()
        if p is not None and pd.notna(p)
    }

    # (col, window_size) — None means use full buffer
    _vel_vol_specs = [
        ("price_velocity",       None),
        ("price_volatility",     None),
        ("price_velocity_1m",    1),
        ("price_volatility_1m",  1),
        ("price_velocity_3m",    3),
        ("price_volatility_3m",  3),
        ("price_velocity_6m",    6),
        ("price_volatility_6m",  6),
        ("price_velocity_24m",   24),
        ("price_volatility_24m", 24),
        ("price_velocity_60m",   60),
        ("price_volatility_60m", 60),
    ]

    all_months  = pd.date_range(start_date, end=end_date, freq="MS")
    all_records = []

    for month_dt in all_months:
        active_ids = [gid for gid, gs in game_starts.items() if gs <= month_dt]
        if not active_ids:
            continue
        active_df = df_base[df_base["igdb_id"].isin(set(active_ids))].copy()

        for gid in active_ids:
            months_counter[gid] = months_counter.get(gid, 0) + 1

        ids = active_df["igdb_id"].tolist()
        ma_vals = [months_counter.get(gid, 1) for gid in ids]
        ma_series = pd.Series(ma_vals, index=active_df.index)

        active_df["prediction_date"] = month_dt.strftime("%Y-%m-01")
        active_df["months_ahead"]    = ma_vals
        active_df["prev_real_price"] = [prev_price.get(gid) for gid in ids]

        # Update current_real_price to the latest predicted price so the model
        # knows the current price level rather than seeing a frozen snapshot.
        active_df["current_real_price"] = [prev_price.get(gid) for gid in ids]

        # Update velocity/volatility from rolling prediction buffer so the model
        # sees the developing price trend rather than static historical metrics.
        for feat, window in _vel_vol_specs:
            if feat not in active_df.columns:
                continue
            is_vel = "velocity" in feat
            vals = []
            for gid in ids:
                buf = predicted_buffer.get(gid, [])
                sliced = buf[-window:] if window else buf
                v = _buf_velocity(sliced) if is_vel else _buf_volatility(sliced)
                vals.append(v)
            updates = pd.Series(vals, index=active_df.index)
            active_df[feat] = updates.where(updates.notna(), active_df[feat])

        # Update absolute and percentage change features from buffer.
        _change_specs = [
            ("price_change_1m",      2,  False),
            ("price_change_3m",      4,  False),
            ("price_change_6m",      7,  False),
            ("price_change_12m",     13, False),
            ("price_pct_change_1m",  2,  True),
            ("price_pct_change_3m",  4,  True),
            ("price_pct_change_6m",  7,  True),
            ("price_pct_change_12m", 13, True),
        ]
        for feat, n, pct in _change_specs:
            if feat not in active_df.columns:
                continue
            vals = []
            for gid in ids:
                buf = predicted_buffer.get(gid, [])
                if len(buf) >= n:
                    delta = buf[-1] - buf[-n]
                    vals.append(round(delta / buf[-n], 4) if pct and buf[-n] > 0 else round(delta, 4))
                else:
                    vals.append(None)
            updates = pd.Series(vals, index=active_df.index)
            active_df[feat] = updates.where(updates.notna(), active_df[feat])

        # Update 52-week high/low ratios.
        for feat, agg_fn in [("price_vs_52w_high", max), ("price_vs_52w_low", min)]:
            if feat not in active_df.columns:
                continue
            vals = []
            for gid in ids:
                buf = predicted_buffer.get(gid, [])
                if len(buf) >= 2:
                    extreme = agg_fn(buf[-12:])
                    vals.append(round(buf[-1] / extreme, 4) if extreme > 0 else None)
                else:
                    vals.append(None)
            updates = pd.Series(vals, index=active_df.index)
            active_df[feat] = updates.where(updates.notna(), active_df[feat])

        # Update price acceleration (change in velocity over rolling buffer halves).
        if "price_acceleration" in active_df.columns:
            vals = []
            for gid in ids:
                buf = predicted_buffer.get(gid, [])
                if len(buf) >= 4:
                    mid   = len(buf) // 2
                    v_rec = _buf_velocity(buf[mid:])
                    v_old = _buf_velocity(buf[:mid])
                    vals.append(round(v_rec - v_old, 4) if v_rec is not None and v_old is not None else None)
                else:
                    vals.append(None)
            updates = pd.Series(vals, index=active_df.index)
            active_df["price_acceleration"] = updates.where(updates.notna(), active_df["price_acceleration"])

        # Project market-wide and genre-level price indices forward.
        # This updates price_vs_market_mean, genre_median_price, and price_vs_genre
        # so the model sees a plausible market context rather than a frozen snapshot.
        mi = market_proj.get(month_dt)
        if mi and mi > 0 and "price_vs_market_mean" in active_df.columns:
            active_df["price_vs_market_mean"] = [
                round(prev_price.get(gid, 0) / mi, 4) if prev_price.get(gid) else None
                for gid in ids
            ]

        if genre_proj and "genre_median_price" in active_df.columns:
            gmed_vals, pvg_vals = [], []
            for i, gid in enumerate(ids):
                g    = active_df.iloc[i].get("genre") if "genre" in active_df.columns else None
                gmed = genre_proj.get(str(g), {}).get(month_dt) if g else None
                p    = prev_price.get(gid)
                gmed_vals.append(round(gmed, 2) if gmed else None)
                pvg_vals.append(round(p / gmed, 4) if (p and gmed and gmed > 0) else None)
            gmed_s = pd.Series(gmed_vals, index=active_df.index)
            pvg_s  = pd.Series(pvg_vals,  index=active_df.index)
            active_df["genre_median_price"] = gmed_s.where(gmed_s.notna(), active_df["genre_median_price"])
            if "price_vs_genre" in active_df.columns:
                active_df["price_vs_genre"] = pvg_s.where(pvg_s.notna(), active_df["price_vs_genre"])

        for col in ["months_since_release", "months_since_discontinued"]:
            if col in base_tv:
                bv = pd.Series([base_tv[col].get(gid) for gid in ids], index=active_df.index)
                active_df[col] = (bv + ma_series).where(bv.notna())

        if "days_since_last_franchise_release" in base_tv:
            bv = pd.Series([base_tv["days_since_last_franchise_release"].get(gid) for gid in ids], index=active_df.index)
            active_df["days_since_last_franchise_release"] = (bv + ma_series * days_per_month).where(bv.notna())

        if "days_until_next_franchise_release" in base_tv:
            bv = pd.Series([base_tv["days_until_next_franchise_release"].get(gid) for gid in ids], index=active_df.index)
            active_df["days_until_next_franchise_release"] = (bv - ma_series * days_per_month).clip(lower=0.0).where(bv.notna())

        active_df["publisher_encoded"] = enc.transform(
            active_df[["publisher"]].fillna("Unknown")
        ).ravel()

        X_mid = prepare_features(active_df)
        def _safe_prev(d, gid):
            v = d.get(gid)
            return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        prev_arr = np.array([_safe_prev(prev_price, gid) for gid in ids])

        # Models predict monthly log-return. Compound only the median path.
        delta_mid     = models["median"].predict(X_mid)
        delta_q10_1st = np.minimum(models["lower"].predict(X_mid), delta_mid)
        delta_q90_1st = np.maximum(models["upper"].predict(X_mid), delta_mid)

        # Widen band with time since last known price — further out = more uncertain
        ma_arr        = np.array([months_counter.get(gid, 1) for gid in ids], dtype=float)
        half_interval = (delta_q90_1st - delta_q10_1st) / 2.0 * np.sqrt(ma_arr)

        pred_abs = prev_arr * np.exp(delta_mid)
        q10_abs  = pred_abs * np.exp(-half_interval)
        q90_abs  = pred_abs * np.exp(+half_interval)

        for gid, p in zip(ids, pred_abs):
            prev_price[gid] = float(p)
            buf = predicted_buffer.setdefault(gid, [])
            buf.append(float(p))

        last_known = active_df.get("current_real_price", pd.Series(np.nan, index=active_df.index)).values
        base_cols  = active_df[["igdb_id", "title", "console", "release_year",
                                 "months_ahead", "prediction_date"]].copy()
        all_records.append(build_prediction_df(base_cols, pred_abs, q10_abs, q90_abs, last_known_price=last_known))

    return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()

def run_eval_phase():
    yr1        = _cutoff_year + 1
    yr2        = _cutoff_year + 2
    eval_start = pd.Timestamp(f"{yr1}-01-01")

    if not EVAL_INPUT.exists():
        return None

    df = pd.read_csv(EVAL_INPUT)

    if "target_months_json" not in df.columns:
        return None

    df = df.dropna(subset=["target_real_price", "release_year"])
    df["release_year"] = df["release_year"].astype(int)

    cpi = load_cpi_for_model()

    eval_rows    = expand_eval_rows(df)
    eval_rows    = eval_rows.dropna(subset=[TARGET])
    df_train_all = build_training_data(df, cpi, eval_start, extra_rows=eval_rows)

    if len(df_train_all) < 100:
        return None

    models, enc, feature_names = train_phase(df_train_all, "eval")

    eval_end      = eval_start + pd.DateOffset(months=35)
    external_proj = compute_external_projections(df, cpi, eval_start, eval_end)
    out = sequential_inference(df, models, enc, eval_start, eval_end,
                               per_game_start=False, external_proj=external_proj)

    # Join actual absolute prices (not deltas — TARGET is now a delta)
    if not eval_rows.empty and "actual_real_price" in eval_rows.columns:
        actual_map = (
            eval_rows.drop_duplicates(["igdb_id", "months_ahead"])
                     .set_index(["igdb_id", "months_ahead"])["actual_real_price"]
        )
        out["actual_price"] = out.apply(
            lambda r: round(actual_map.get((r["igdb_id"], r["months_ahead"]), np.nan), 2), axis=1
        )

    actual_prices = out["actual_price"].values if "actual_price" in out.columns else np.full(len(out), np.nan)
    pred_mid      = out["prediction"].values
    q10           = out["lower_bound"].values
    q90           = out["upper_bound"].values

    eval_preds_path = DATA_DIR / f"eval_predictions_{_cutoff_year}.csv"
    out.to_csv(eval_preds_path, index=False)
    print(f"Saved to {eval_preds_path}")

    baseline  = out.get("real_price_at_60m", pd.Series(np.nan, index=out.index)).values
    mask_1yr  = out["months_ahead"].between(1, 12).values
    mask_2yr  = out["months_ahead"].between(13, 24).values

    m1 = eval_horizon_metrics(
        actual_prices[mask_1yr], pred_mid[mask_1yr], q10[mask_1yr], q90[mask_1yr],
        baseline[mask_1yr],
        f"1yr horizon (months 1–12, Jan–Dec {yr1})"
    )
    m2 = eval_horizon_metrics(
        actual_prices[mask_2yr], pred_mid[mask_2yr], q10[mask_2yr], q90[mask_2yr],
        baseline[mask_2yr],
        f"2yr horizon (months 13–24, Jan–Dec {yr2})"
    )

    eval_model_dir = MODELS_DIR / "eval"
    eval_model_dir.mkdir(parents=True, exist_ok=True)
    for lbl, model in models.items():
        model.save_model(str(eval_model_dir / f"lgb_{lbl}.txt"))
    joblib.dump(enc, eval_model_dir / "publisher_encoder.pkl")
    print(f"Saved eval models to {eval_model_dir}/")

    return {
        "description": f"Trained ≤Dec {_cutoff_year}, backtested Jan {yr1}–Dec {yr2}",
        "n_games":     int(out["igdb_id"].nunique()),
        "n_rows":      len(out),
        "horizon_1yr": m1,
        "horizon_2yr": m2,
    }

def run_prod_phase():
    n_months = PROD_FORECAST_MONTHS
    end_str  = PROD_FORECAST_END.strftime("%b %Y")

    df = pd.read_csv(PROD_INPUT)
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    cpi = load_cpi_for_model()

    extra_rows = pd.DataFrame()
    if EVAL_INPUT.exists():
        df_eval = pd.read_csv(EVAL_INPUT)
        if "target_months_json" in df_eval.columns:
            df_eval = df_eval.dropna(subset=["target_real_price", "release_year"])
            df_eval["release_year"] = df_eval["release_year"].astype(int)
            extra_rows = expand_eval_rows(df_eval)

    df_train_all = build_training_data(df, cpi, PROD_FORECAST_START, extra_rows=extra_rows)

    if df_train_all.empty:
        return None

    # 10-fold CV for per-game error weights — commented out for fast iteration
    # if EVAL_INPUT.exists():
    #     df_eval_kf = pd.read_csv(EVAL_INPUT)
    #     if "target_months_json" in df_eval_kf.columns:
    #         df_eval_kf = df_eval_kf.dropna(subset=["target_real_price", "release_year"])
    #         df_eval_kf["release_year"] = df_eval_kf["release_year"].astype(int)
    #         print("Running 10-fold CV for sample weights…")
    #         sample_weights = kfold_error_weights(df_eval_kf, cpi, pd.Timestamp("2024-01-01"))
    sample_weights = None

    models, enc, feature_names = train_phase(df_train_all, "prod", sample_weights=sample_weights)

    external_proj = compute_external_projections(df, cpi, PROD_FORECAST_START, PROD_FORECAST_END)
    out = sequential_inference(df, models, enc, PROD_FORECAST_START, PROD_FORECAST_END,
                               per_game_start=True, external_proj=external_proj)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PREDS, index=False)
    print(f"Saved to {OUTPUT_PREDS}")

    prod_model_dir = MODELS_DIR / "prod"
    prod_model_dir.mkdir(parents=True, exist_ok=True)
    for lbl, model in models.items():
        model.save_model(str(prod_model_dir / f"lgb_{lbl}.txt"))
    joblib.dump(enc, prod_model_dir / "publisher_encoder.pkl")
    print(f"Saved prod models to {prod_model_dir}/")

    # SHAP: snapshot at months_ahead=12 using base dataset features
    df_shap = df.copy()
    df_shap["months_ahead"]    = 12
    df_shap["prev_real_price"] = df_shap.get("current_real_price", pd.Series(np.nan, index=df_shap.index))
    for col in ["months_since_release", "months_since_discontinued"]:
        if col in df_shap.columns:
            df_shap[col] = df_shap[col].fillna(0) + 12
    df_shap["publisher_encoded"] = enc.transform(df_shap[["publisher"]].fillna("Unknown")).ravel()
    X_shap_all  = prepare_features(df_shap)
    shap_sample = X_shap_all.sample(min(2000, len(X_shap_all)), random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(models["median"])
    shap_values = explainer.shap_values(shap_sample)
    shap_df     = pd.DataFrame(shap_values, columns=feature_names)
    igdb_ids    = df_shap.reset_index(drop=True).loc[shap_sample.index, "igdb_id"].values
    shap_df.insert(0, "igdb_id", igdb_ids)
    shap_df.to_csv(OUTPUT_SHAP, index=False)
    print(f"Saved to {OUTPUT_SHAP}")

    mean_conf = out["confidence_pct"].mean()
    conf_12mo = out[out["months_ahead"] == 12]["confidence_pct"].mean()
    conf_60mo = out[out["months_ahead"] == 60]["confidence_pct"].mean() if n_months >= 60 else np.nan

    return {
        "description":         f"Jan 2025–{end_str} ({n_months} months per game)",
        "n_games":             out["igdb_id"].nunique(),
        "n_prediction_rows":   len(out),
        "mean_confidence_pct": round(float(mean_conf), 1),
        "confidence_at_12mo":  round(float(conf_12mo), 1),
        "confidence_at_60mo":  round(float(conf_60mo), 1) if not np.isnan(conf_60mo) else None,
    }

def main():
    eval_metrics = run_eval_phase()
    prod_metrics = run_prod_phase()

    metrics = {
        "eval_model": eval_metrics or {},
        "prod_model": prod_metrics or {},
    }
    with open(OUTPUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()
