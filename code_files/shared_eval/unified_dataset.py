"""Unified dataset layer.

Loads raw OptionMetrics CSVs, runs the shared preprocessing + Moussa
arbitrage filter, BS-inverts every arb-filtered mid-price into a real
market IV, applies one global quality filter, and splits the dates into
TRAIN / VAL / TEST by year (2013-17 / 2018 / 2019).

The output is THE single source of truth for the cross-pipeline study:
- `quotes_dict[date]`: per-day dataframe of (k, T, sigma_market, ...) tuples.
- `splits`: dict with date lists for train / val / test.
- `side_info`: per-day exogenous features used by VolGAN
  (logret_{t-1}, logret_{t-2}, RV21_{t-1}).

No pipeline may re-filter or re-split this data.

Cache file: data/unified/quotes.pkl
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code_files") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code_files"))


# ---------------------------------------------------------------------------
# Default unified filter (the §3 quality bar that every pipeline sees)
# ---------------------------------------------------------------------------

T_MIN_DAYS = 7
T_MAX_DAYS = 730            # 2y, matches D/E
MONEYNESS_LO = 0.7
MONEYNESS_HI = 1.3
IV_LO = 0.03
IV_HI = 1.50
PRICE_MIN = 0.05
MIN_QUOTES_PER_DAY = 30

# Year-based split (matches §4 of plan.md).
TRAIN_YEARS = (2013, 2014, 2015, 2016, 2017)
VAL_YEARS = (2018,)
TEST_YEARS = (2019,)


# ---------------------------------------------------------------------------
# Black-Scholes IV inversion (forward-measure Black-76, consistent with
# pipeline E's build_cache.py)
# ---------------------------------------------------------------------------

def _black76_call(F, K, T, sigma, disc):
    """European call under Black-76 (forward measure)."""
    F = float(F); K = float(K); T = float(T); sigma = float(sigma); disc = float(disc)
    if T <= 0 or sigma <= 0:
        return disc * max(F - K, 0.0)
    sigT = sigma * np.sqrt(T)
    d1 = (np.log(max(F, 1e-12) / max(K, 1e-12)) + 0.5 * sigma * sigma * T) / sigT
    d2 = d1 - sigT
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _bs_implied_vol(price, F, K, T, disc):
    """Brent-bracketed IV inversion. Returns NaN on failure."""
    if not (price > 0 and F > 0 and K > 0 and T > 0 and disc > 0):
        return np.nan
    intrinsic = disc * max(F - K, 0.0)
    if price <= intrinsic + 1e-10:
        return np.nan
    upper_bound = disc * F  # call price < disc * F
    if price >= upper_bound - 1e-10:
        return np.nan
    f = lambda s: _black76_call(F, K, T, s, disc) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# Build / load the unified cache
# ---------------------------------------------------------------------------

def _split_year(date):
    y = pd.Timestamp(date).year
    if y in TRAIN_YEARS:
        return "train"
    if y in VAL_YEARS:
        return "val"
    if y in TEST_YEARS:
        return "test"
    return None


def build_unified_dataset(
    *,
    csv_dir: Path | str = "../data_csv",
    out_path: Path | str = "data/unified/quotes.pkl",
    moussa_filtered_pkl: Path | str = "data/shared/filtered_dict.pkl",
    market_data_pkl: Path | str = "data/shared/market_data.pkl",
    rebuild_moussa: bool = False,
    verbose: bool = True,
):
    """Build the unified per-quote dataset and cache it.

    Reuses the existing Moussa-filtered cache from `data/shared/filtered_dict.pkl`
    if available (built by Pipeline A's stage 1+2). Set `rebuild_moussa=True`
    to redo from scratch.
    """
    csv_dir = (PROJECT_ROOT / csv_dir) if not Path(csv_dir).is_absolute() else Path(csv_dir)
    out_path = (PROJECT_ROOT / out_path) if not Path(out_path).is_absolute() else Path(out_path)
    moussa_filtered_pkl = (PROJECT_ROOT / moussa_filtered_pkl) if not Path(moussa_filtered_pkl).is_absolute() else Path(moussa_filtered_pkl)
    market_data_pkl = (PROJECT_ROOT / market_data_pkl) if not Path(market_data_pkl).is_absolute() else Path(market_data_pkl)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("BUILDING UNIFIED DATASET")
        print("=" * 60)

    # -- 1. Get the Moussa-filtered surface dict ---------------------------
    if rebuild_moussa or not moussa_filtered_pkl.exists():
        from shared.pipeline_part1_preprocessing import load_all_data, preprocess
        from shared.pipeline_part2_moussa_filter import moussa_filter_surface

        if verbose:
            print(f"\n[1/4] Loading raw CSVs from {csv_dir} ...")
        # The legacy preprocessor expects bare filenames without .csv extension.
        # We adapt by symlinking/renaming if needed; here we just call it with
        # the directory path and trust the existing convention.
        # Falling back to direct CSV load:
        data = _load_raw_csvs(csv_dir)
        if verbose:
            print(f"\n[2/4] Preprocessing + Moussa filter ...")
        surface_dict = preprocess(data, min_volume=10)
        filtered_dict, _stats = moussa_filter_surface(surface_dict)
        # Save market data alongside
        market_data = {
            "St_df": data["St_df"],
            "fwd_df": data["fwd_df"],
            "discount_df": data["discount_df"],
        }
        moussa_filtered_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(moussa_filtered_pkl, "wb") as f:
            pickle.dump(filtered_dict, f)
        with open(market_data_pkl, "wb") as f:
            pickle.dump(market_data, f)
    else:
        if verbose:
            print(f"\n[1/4] Loading cached Moussa-filtered dict from "
                  f"{moussa_filtered_pkl} ...")
        with open(moussa_filtered_pkl, "rb") as f:
            filtered_dict = pickle.load(f)
        with open(market_data_pkl, "rb") as f:
            market_data = pickle.load(f)

    if verbose:
        print(f"  filtered_dict has {len(filtered_dict)} dates")

    # -- 2. Build per-quote dataset, BS-inverting to IV --------------------
    if verbose:
        print(f"\n[3/4] Building per-quote dataset (BS-inversion + filters) ...")

    quotes_dict: dict[pd.Timestamp, pd.DataFrame] = {}
    n_in = 0
    n_out = 0

    for d, df in filtered_dict.items():
        d_norm = pd.Timestamp(d).normalize()
        df = df.copy()
        n_in += len(df)

        # Tau (years)
        if "maturity_days" not in df.columns and "exdate" in df.columns:
            df["maturity_days"] = (
                pd.to_datetime(df["exdate"]) - pd.to_datetime(d_norm)
            ).dt.days
        df["T"] = df["maturity_days"] / 365.0

        # Forward / discount (already present from preprocessing)
        if "fwd_price" not in df.columns or "discount" not in df.columns:
            continue

        # k = log(K / F)
        df["log_moneyness"] = np.log(df["strike"] / df["fwd_price"])
        df["mny"] = df["strike"] / df["fwd_price"]

        # Use moussa_price (arb-cleaned) when present, else midP
        if "moussa_price" in df.columns:
            df["mid"] = df["moussa_price"]
        else:
            df["mid"] = df["midP"]

        # Day-level pre-filter
        mask = (
            (df["maturity_days"] >= T_MIN_DAYS)
            & (df["maturity_days"] <= T_MAX_DAYS)
            & (df["mny"] >= MONEYNESS_LO)
            & (df["mny"] <= MONEYNESS_HI)
            & (df["mid"] >= PRICE_MIN)
            & (df["fwd_price"] > 0)
            & (df["discount"] > 0)
        )
        df = df[mask].copy()
        if len(df) < MIN_QUOTES_PER_DAY:
            continue

        # BS-invert each row to get sigma_market.
        ivs = np.empty(len(df), dtype=np.float64)
        for i, row in enumerate(df.itertuples(index=False)):
            ivs[i] = _bs_implied_vol(
                price=row.mid,
                F=row.fwd_price,
                K=row.strike,
                T=row.T,
                disc=row.discount,
            )
        df["sigma_market"] = ivs

        # Drop failed inversions and IV-bound violations.
        df = df[
            np.isfinite(df["sigma_market"])
            & (df["sigma_market"] >= IV_LO)
            & (df["sigma_market"] <= IV_HI)
        ].copy()

        if len(df) < MIN_QUOTES_PER_DAY:
            continue

        # Keep only the columns we need downstream.
        df_out = df[[
            "strike", "exdate", "maturity_days", "T",
            "fwd_price", "discount",
            "mid", "log_moneyness", "mny", "sigma_market",
        ]].reset_index(drop=True)

        quotes_dict[d_norm] = df_out
        n_out += len(df_out)

    if verbose:
        print(f"  {n_in} input rows -> {n_out} arb-filtered+BS-inverted quotes")
        print(f"  Days kept: {len(quotes_dict)}")

    # -- 3. Splits + side-info --------------------------------------------
    if verbose:
        print(f"\n[4/4] Splits + side-info ...")
    dates_sorted = sorted(quotes_dict.keys())
    splits = {"train": [], "val": [], "test": []}
    for d in dates_sorted:
        s = _split_year(d)
        if s is not None:
            splits[s].append(d)

    if verbose:
        for s in ("train", "val", "test"):
            print(f"  {s}: {len(splits[s])} dates")

    side_info = _build_side_info(market_data["St_df"], dates_sorted)

    payload = {
        "quotes_dict": quotes_dict,
        "splits": splits,
        "side_info": side_info,
        "market_data": market_data,
        "filter_config": {
            "T_min_days": T_MIN_DAYS, "T_max_days": T_MAX_DAYS,
            "moneyness_lo": MONEYNESS_LO, "moneyness_hi": MONEYNESS_HI,
            "iv_lo": IV_LO, "iv_hi": IV_HI,
            "price_min": PRICE_MIN,
            "min_quotes_per_day": MIN_QUOTES_PER_DAY,
            "train_years": TRAIN_YEARS, "val_years": VAL_YEARS,
            "test_years": TEST_YEARS,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"\n  Saved unified cache: {out_path}")

    return payload


def load_unified_dataset(
    *,
    out_path: Path | str = "data/unified/quotes.pkl",
    rebuild: bool = False,
    **build_kwargs,
):
    """Return the cached unified dataset; build it if missing or rebuild=True."""
    out_path = (PROJECT_ROOT / out_path) if not Path(out_path).is_absolute() else Path(out_path)
    if rebuild or not out_path.exists():
        return build_unified_dataset(out_path=out_path, **build_kwargs)
    with open(out_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Side info: shared exogenous features for VolGAN
# ---------------------------------------------------------------------------

def _build_side_info(St_df: pd.DataFrame, dates_sorted) -> pd.DataFrame:
    """Build (logret_{t-1}, logret_{t-2}, RV21_{t-1}) per date.

    We use the Spot column from St_df. RV21 is the 21-day annualised realised
    volatility of daily log-returns.
    """
    s = St_df.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s.sort_index()
    if "Spot" in s.columns:
        spot = s["Spot"].astype(float)
    else:
        # First numeric column is the spot.
        spot = s.iloc[:, 0].astype(float)
    logret = np.log(spot / spot.shift(1))
    rv21 = (logret.rolling(21).std() * np.sqrt(252)).shift(1)
    side = pd.DataFrame({
        "logret_lag1": logret.shift(1),
        "logret_lag2": logret.shift(2),
        "rv21_lag1": rv21,
    })
    side = side.reindex(pd.Index(dates_sorted, name="date"))
    return side


# ---------------------------------------------------------------------------
# Raw CSV fallback loader (handles either bare filenames or .csv extensions)
# ---------------------------------------------------------------------------

def _load_raw_csvs(csv_dir: Path) -> dict:
    """Robust loader that handles both `St_df` and `St_df.csv` filenames."""
    def _find(name: str) -> Path:
        for cand in (csv_dir / name, csv_dir / f"{name}.csv"):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Could not find {name} or {name}.csv in {csv_dir}")

    list_exp = pd.read_csv(_find("list_exp"), index_col=0)
    list_exp.columns = ["Days"]

    list_mny = pd.read_csv(_find("list_mny"), index_col=0)
    list_mny.columns = ["Moneyness"]

    St_df = pd.read_csv(_find("St_df"), index_col=0, parse_dates=True)
    fwd_df = pd.read_csv(_find("fwd_df"), index_col=0, parse_dates=True)
    discount_df = pd.read_csv(_find("discount_df"), index_col=0, parse_dates=True)

    op_df = pd.read_csv(_find("op_df"), index_col=0, low_memory=False)
    op_df["date"] = pd.to_datetime(op_df["date"])
    return {
        "op_df": op_df, "fwd_df": fwd_df, "discount_df": discount_df,
        "St_df": St_df, "list_exp": list_exp, "list_mny": list_mny,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build unified dataset cache.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if cache exists.")
    parser.add_argument("--rebuild-moussa", action="store_true",
                        help="Rebuild from raw CSVs (slow).")
    args = parser.parse_args()
    build_unified_dataset(rebuild_moussa=args.rebuild_moussa) \
        if args.rebuild else load_unified_dataset(rebuild=args.rebuild)
