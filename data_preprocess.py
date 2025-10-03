from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

RAW_DIR = Path("data_raw")
OUT_DIR = Path("data_preprocessed")

# ------------------------- Helpers ------------------------- #

def _std_cols(cols):
    return [str(c).strip().lower().replace(" ", "_") for c in cols]


def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _infer_freq(idx: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if idx.size < 3:
        return None
    # Try pandas' infer_freq first
    try:
        freq_str = pd.infer_freq(idx)
        if freq_str is not None:
            return pd.tseries.frequencies.to_offset(freq_str).delta
    except Exception:
        pass
    # Fallback: mode of diffs
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return None
    mode = diffs.mode()
    return mode.iloc[0] if not mode.empty else None


def _drop_bad_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # Basic logical consistency checks
    mask_bad = (
        (df["high"] < df["low"]) |
        (df["high"] < df[["open", "close"]].min(axis=1)) |
        (df["low"] > df[["open", "close"]].max(axis=1))
    )
    if mask_bad.any():
        df = df.loc[~mask_bad].copy()
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Price levels
    df["mid"] = (h + l) / 2.0
    df["hl2"] = (h + l) / 2.0
    df["hlc3"] = (h + l + c) / 3.0
    df["ohlc4"] = (o + h + l + c) / 4.0

    # Candle geometry
    df["range"] = (h - l).abs()
    df["body"] = (c - o).abs()
    df["upper_wick"] = (h - np.maximum(o, c)).clip(lower=0)
    df["lower_wick"] = (np.minimum(o, c) - l).clip(lower=0)
    df["body_to_range"] = (df["body"] / df["range"]).replace([np.inf, -np.inf], np.nan)

    # Returns
    df["ret_1"] = c.pct_change()
    df["log_ret"] = np.log(c).diff()

    # Rolling stats (use safe rolling with min_periods)
    for w in (5, 10, 20, 50, 100):
        df[f"ema_{w}"] = _ema(c, span=w)
        df[f"zclose_{w}"] = _zscore(c, window=w)
        df[f"vol_{w}"] = df["log_ret"].rolling(w, min_periods=max(2, w // 2)).std()

    # ATR (classic Wilder)
    df["atr_14"] = _atr(o, h, l, c, period=14)

    # RSI (Wilder)
    df["rsi_14"] = _rsi(c, period=14)

    # Bollinger (20, 2)
    ma20 = c.rolling(20, min_periods=10).mean()
    sd20 = c.rolling(20, min_periods=10).std()
    df["bb_mid_20"] = ma20
    df["bb_up_20"] = ma20 + 2 * sd20
    df["bb_dn_20"] = ma20 - 2 * sd20
    df["bb_width_20"] = (df["bb_up_20"] - df["bb_dn_20"]) / df["bb_mid_20"].abs()

    # Robust z-score via MAD (good for anomaly work)
    df["rz_close_50"] = _robust_z(c, window=50)

    return df


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def _zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=max(2, window // 2)).mean()
    s = x.rolling(window, min_periods=max(2, window // 2)).std()
    return (x - m) / s


def _robust_z(x: pd.Series, window: int) -> pd.Series:
    med = x.rolling(window, min_periods=max(2, window // 2)).median()
    mad = (x.rolling(window, min_periods=max(2, window // 2)).apply(lambda s: np.median(np.abs(s - np.median(s))), raw=True))
    # 1.4826 ~ consistency constant for normal dist
    return (x - med) / (1.4826 * mad.replace(0, np.nan))


def _atr(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period//2).mean()


def _rsi(x: pd.Series, period: int = 14) -> pd.Series:
    dx = x.diff()
    gain = dx.clip(lower=0.0)
    loss = -dx.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period//2).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period//2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ------------------------- Core Pipeline ------------------------- #

def preprocess_df(df: pd.DataFrame, *, drop_large_jumps: bool = True, jump_factor: float = 5.0) -> Tuple[pd.DataFrame, Optional[pd.Timedelta]]:
    """Clean & enrich an OHLCV DataFrame. Returns (df, inferred_freq).

    Parameters
    ----------
    drop_large_jumps : if True, removes rows immediately after unusually large time jumps
        (delta > jump_factor * inferred_freq). This helps eliminate weekend/holiday gaps
        from downstream anomaly models that assume quasi-regular sampling.
    jump_factor : factor applied on inferred frequency to define a "large" jump.
    """
    # Standardize columns
    df = df.copy()
    df.columns = _std_cols(df.columns)

    # Common MT5 column fix-ups
    rename_map = {
        "time": "datetime",
        "\ufefftime": "datetime",
        "\ufeffdatetime": "datetime",
        "tick_volume": "volume",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Drop real_volume if present per user request
    if "real_volume" in df.columns:
        df = df.drop(columns=["real_volume"])  # explicit instruction

    # Ensure datetime column
    dt_col = None
    for cand in ("datetime", "date", "timestamp"):
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ValueError("No datetime column found. Expected one of: datetime/date/timestamp/time")

    # Parse dates robustly
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    df = df.dropna(subset=[dt_col]).copy()
    df = df.sort_values(dt_col).drop_duplicates(subset=[dt_col])
    df = df.set_index(dt_col)

    # Coerce numeric columns
    for price_col in ["open", "high", "low", "close", "volume", "spread"]:
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Replace infs with NaN and handle missing
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are completely NaN across core OHLC
    core = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if not core:
        raise ValueError("Missing OHLC columns; need at least open/high/low/close")
    df = df.dropna(subset=core, how="any")  # strict: must have all core prices

    # Basic OHLC validation
    df = _coerce_numeric(df, core + (["volume", "spread"] if "volume" in df.columns or "spread" in df.columns else []))
    df = _drop_bad_ohlc(df)

    # Frequency inference & gap flagging
    inferred = _infer_freq(df.index)
    if inferred is not None and len(df) > 1:
        deltas = df.index.to_series().diff()
        df["is_gap_after_prev"] = deltas > inferred
        # Optionally drop rows after "very large" jumps (e.g., weekends)
        if drop_large_jumps:
            large_jump = deltas > (jump_factor * inferred)
            if large_jump.any():
                df = df.loc[~large_jump].copy()
                df["is_gap_after_prev"] = df.index.to_series().diff() > inferred
    else:
        df["is_gap_after_prev"] = False

    # Fill minor NaNs conservatively
    # Prices: forward-fill then back-fill tiny early holes; Volume/spread: fillna(0)
    for col in core:
        df[col] = df[col].ffill().bfill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)
    if "spread" in df.columns:
        df["spread"] = df["spread"].ffill().fillna(0)

    # Feature engineering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df = _add_features(df)

    # Final tidy columns order
    preferred = ["open", "high", "low", "close", "volume"]
    others = [c for c in df.columns if c not in preferred]
    df = df[preferred + others]

    # Remove helper flags by default; keep for debugging if needed
    df = df.drop(columns=["is_gap_after_prev"], errors="ignore")

    return df, inferred


# ------------------------- I/O & CLI ------------------------- #

def run(input_name: str, *, tz: Optional[str] = None, to_utc: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = RAW_DIR / input_name
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Load CSV (robust parser)
    df = pd.read_csv(in_path)

    # Preprocess
    df_clean, inferred = preprocess_df(df)

    # Optional TZ localization/conversion
    if tz is not None:
        try:
            df_clean.index = df_clean.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        except TypeError:
            # If already tz-aware, convert
            df_clean.index = df_clean.index.tz_convert(tz)
    if to_utc and df_clean.index.tz is not None:
        df_clean.index = df_clean.index.tz_convert("UTC")

    # Output filenames
    base = in_path.stem
    out_csv = OUT_DIR / f"{base}__preprocessed.csv"
    out_parquet = OUT_DIR / f"{base}__preprocessed.parquet"

    # Save
    df_clean.to_csv(out_csv, index_label="datetime")
    try:
        df_clean.to_parquet(out_parquet, index=True)
    except Exception as e:
        # Parquet optional; keep going if missing engine
        print(f"[WARN] Could not save Parquet: {e}")

    # Inform about inferred frequency
    if inferred is not None:
        print(f"Inferred frequency: {inferred}")
    print(f"Saved: {out_csv}")
    if out_parquet.exists():
        print(f"Saved: {out_parquet}")

    return out_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess OHLCV CSV from data_raw to data_preprocessed with robust cleaning and features.")
    p.add_argument("--input",default="FUNDEDNEXT_EURUSD_15_2000-01-01_2025-01-01.csv", help="CSV filename inside data_raw/")
    p.add_argument("--tz", default=None, help="Timezone to localize/convert index (e.g., 'UTC', 'Europe/Madrid').")
    p.add_argument("--to-utc", action="store_true", help="If set, convert timezone-aware index to UTC after --tz.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, tz=args.tz, to_utc=args.to_utc)
