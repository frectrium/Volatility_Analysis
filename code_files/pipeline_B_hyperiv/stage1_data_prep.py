"""
=============================================================================
PIPELINE — HyperIV Data Preparation
=============================================================================
Adapts the existing SPX EOD preprocessed data (from Stages 1-2 of the main
pipeline) into the format required by HyperIV training.

Input:  Moussa-filtered surface_dict from Stage 2
Output: DataFrame with columns needed for HyperIV, including:
        - date, log_moneyness, tau, implied_volatility, delta, is_ref
        - forward_price, risk_free_rate, strike_price, option_price

The key adaptation is:
  1. Compute implied volatility via BS inversion (HyperIV needs sigma, not prices)
  2. Compute delta for each contract
  3. Select 9 reference contracts per date: {ATM, 25D Call, 25D Put} x {7d, 30d, 90d}
  4. Mark reference contracts with is_ref=1
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import pickle
from pathlib import Path
from tqdm import tqdm


# ============================================================
# 1. BLACK-SCHOLES UTILITIES
# ============================================================
def bs_price_call(K, T, F, sigma, r=0.0):
    """Black-Scholes call price using forward price."""
    if T <= 0 or sigma <= 0:
        return max(F * np.exp(-r * T) - K * np.exp(-r * T), 0.0)
    k = np.log(K / F)
    d1 = (-k + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * F * (norm.cdf(d1) - np.exp(k) * norm.cdf(d2))


def bs_implied_vol(price, K, T, F, r=0.0, is_call=True):
    """
    Invert BS formula to get implied volatility using Brent's method.

    Args:
        price: observed option mid-price
        K: strike
        T: time to maturity in years
        F: forward price
        r: risk-free rate
        is_call: True for calls

    Returns:
        implied volatility or NaN if inversion fails
    """
    if T <= 0 or price <= 0:
        return np.nan

    # For calls: use directly
    # Our data is already calls (from Stage 1 filtering)
    intrinsic = max(np.exp(-r * T) * (F - K), 0.0)
    if price < intrinsic - 1e-6:
        return np.nan

    def objective(sigma):
        return bs_price_call(K, T, F, sigma, r) - price

    try:
        iv = brentq(objective, 1e-4, 5.0, xtol=1e-10, maxiter=200)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def bs_delta(K, T, F, sigma, is_call=1):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 0.0
    k = np.log(K / F)
    d1 = (-k + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    if is_call == 1:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


# ============================================================
# 2. REFERENCE SET SELECTION
# ============================================================
def select_reference_set(date_df):
    """
    Select 9 reference contracts for a single date.

    Target grid: {ATM, 25D Call, 25D Put} x {7-day, 30-day, 90-day}

    For each (delta_target, ttm_target):
      1. Find contracts closest to the target TTM
      2. Among those, find the one closest to the target delta

    Args:
        date_df: DataFrame for one date with columns:
                 maturity_days, delta, log_moneyness, tau, implied_volatility

    Returns:
        ref_indices: list of DataFrame indices for the 9 reference contracts
    """
    target_ttms = [7, 30, 90]
    # ATM ~ 0.5 delta for calls, 25D call ~ 0.25, 25D put ~ -0.25
    # Since we only have calls, ATM is ~0.5, OTM calls have delta < 0.5
    target_deltas = [0.75, 0.5, 0.25]

    ref_indices = []

    for ttm in target_ttms:
        # Find contracts closest to target TTM
        ttm_diff = (date_df["maturity_days"] - ttm).abs()
        # Take contracts within a window, or the closest TTM available
        min_ttm_diff = ttm_diff.min()
        ttm_candidates = date_df[ttm_diff <= max(min_ttm_diff + 3, 5)]

        if len(ttm_candidates) == 0:
            ttm_candidates = date_df.nsmallest(10, "maturity_days")

        for target_delta in target_deltas:
            delta_diff = (ttm_candidates["delta"] - target_delta).abs()
            best_idx = delta_diff.idxmin()
            ref_indices.append(best_idx)

    return ref_indices


# ============================================================
# 3. PROCESS ONE DATE
# ============================================================
def process_date(date_val, date_surface, fwd_interp, disc_interp):
    """
    Process one date's filtered surface data into HyperIV format.

    Args:
        date_val: the date
        date_surface: DataFrame from filtered_dict[date]
        fwd_interp: interpolation function for forward prices
        disc_interp: interpolation function for discount factors

    Returns:
        DataFrame with columns needed for HyperIV, or None if insufficient data
    """
    df = date_surface.copy()

    # Basic columns
    if "maturity_days" not in df.columns and "exdate" in df.columns:
        df["maturity_days"] = (pd.to_datetime(df["exdate"]) -
                               pd.to_datetime(date_val)).dt.days

    df["tau"] = df["maturity_days"] / 365.0

    # Ensure forward prices are available
    if "fwd_price" not in df.columns:
        df["fwd_price"] = fwd_interp(df["maturity_days"].values.astype(float))

    # Ensure discount factors
    if "discount" not in df.columns:
        df["discount"] = disc_interp(df["maturity_days"].values.astype(float))

    # Compute risk-free rate from discount: B(T) = exp(-rT) => r = -ln(B)/T
    df["risk_free_rate"] = -np.log(np.clip(df["discount"], 1e-10, 1.0)) / \
                           np.clip(df["tau"], 1e-6, None)

    # Log-forward moneyness: k = log(K/F)
    df["log_moneyness"] = np.log(df["strike"] / df["fwd_price"])

    # Mid-price (use moussa_price if available, else midP)
    if "moussa_price" in df.columns:
        df["option_price"] = df["moussa_price"]
    elif "midP" in df.columns:
        df["option_price"] = df["midP"]
    else:
        df["option_price"] = (df["best_bid"] + df["best_offer"]) / 2.0

    # Filter out obviously bad data
    df = df[df["option_price"] > 0.01].copy()
    df = df[df["tau"] > 0.001].copy()
    df = df[df["fwd_price"] > 0].copy()

    if len(df) < 9:
        return None

    # Compute implied volatility
    ivs = []
    for _, row in df.iterrows():
        iv = bs_implied_vol(
            price=row["option_price"],
            K=row["strike"],
            T=row["tau"],
            F=row["fwd_price"],
            r=row["risk_free_rate"],
        )
        ivs.append(iv)

    df["implied_volatility"] = ivs

    # Drop NaN IVs
    df = df.dropna(subset=["implied_volatility"]).copy()

    # Filter reasonable IVs
    df = df[(df["implied_volatility"] > 0.01) &
            (df["implied_volatility"] < 3.0)].copy()

    if len(df) < 9:
        return None

    # Compute delta (all calls, is_call=1)
    deltas = []
    for _, row in df.iterrows():
        d = bs_delta(
            K=row["strike"],
            T=row["tau"],
            F=row["fwd_price"],
            sigma=row["implied_volatility"],
            is_call=1,
        )
        deltas.append(d)

    df["delta"] = deltas
    df["is_call"] = 1

    # Strike price column name normalisation
    df["strike_price"] = df["strike"]
    df["forward_price"] = df["fwd_price"]

    # Select reference contracts (deduplicate indices)
    try:
        ref_indices = select_reference_set(df)
        ref_indices = list(dict.fromkeys(ref_indices))  # Deduplicate, preserving order
        df["is_ref"] = 0
        df.loc[ref_indices, "is_ref"] = 1
    except Exception:
        # Fallback: use first 9 contracts
        df["is_ref"] = 0
        df.iloc[:min(9, len(df)), df.columns.get_loc("is_ref")] = 1

    # Add date column
    df["date"] = date_val

    # Time to maturity in days (for reference selection compatibility)
    df["time_to_maturity"] = df["maturity_days"]

    # Select output columns
    out_cols = [
        "date", "strike_price", "forward_price", "tau", "risk_free_rate",
        "is_call", "option_price", "log_moneyness", "implied_volatility",
        "delta", "time_to_maturity", "is_ref",
    ]
    return df[out_cols].reset_index(drop=True)


# ============================================================
# 4. MAIN DATA PREPARATION PIPELINE
# ============================================================
def prepare_hyperiv_data(
    filtered_dict,
    fwd_df,
    discount_df,
    output_path=None,
    min_contracts_per_date=20,
):
    """
    Convert the full filtered surface dictionary to HyperIV training format.

    Args:
        filtered_dict: dict {date: DataFrame} from Stage 2
        fwd_df: DataFrame of forward prices (index=dates, cols=tenors)
        discount_df: DataFrame of discount factors
        output_path: optional path to save the result
        min_contracts_per_date: minimum contracts needed per date

    Returns:
        df_all: DataFrame with all dates concatenated, ready for HyperIV
        stats: dict with processing statistics
    """
    print("\n" + "=" * 60)
    print("HYPERIV DATA PREPARATION")
    print("=" * 60)

    TENORS = np.array([0, 7, 14, 30, 60, 90, 180, 350])

    # Build interpolation maps
    from scipy.interpolate import interp1d

    all_dfs = []
    n_skipped = 0
    n_processed = 0

    sorted_dates = sorted(filtered_dict.keys())
    print(f"  Processing {len(sorted_dates)} dates...")

    for date_val in tqdm(sorted_dates, desc="  Preparing data"):
        # Get forward/discount interpolation for this date
        date_pd = pd.Timestamp(date_val)

        # Find nearest date in fwd_df
        fwd_idx = fwd_df.index.get_indexer([date_pd], method="nearest")[0]
        disc_idx = discount_df.index.get_indexer([date_pd], method="nearest")[0]

        if fwd_idx < 0 or disc_idx < 0:
            n_skipped += 1
            continue

        fwd_vals = fwd_df.iloc[fwd_idx].values.astype(float)
        disc_vals = discount_df.iloc[disc_idx].values.astype(float)

        fwd_interp = interp1d(TENORS, fwd_vals, kind="linear",
                              bounds_error=False,
                              fill_value=(fwd_vals[0], fwd_vals[-1]))
        disc_interp = interp1d(TENORS, disc_vals, kind="linear",
                               bounds_error=False,
                               fill_value=(disc_vals[0], disc_vals[-1]))

        result = process_date(date_val, filtered_dict[date_val],
                              fwd_interp, disc_interp)

        if result is not None and len(result) >= min_contracts_per_date:
            all_dfs.append(result)
            n_processed += 1
        else:
            n_skipped += 1

    if not all_dfs:
        raise ValueError("No dates produced valid HyperIV data!")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Ensure date column is datetime
    df_all["date"] = pd.to_datetime(df_all["date"])

    stats = {
        "n_dates_processed": n_processed,
        "n_dates_skipped": n_skipped,
        "n_total_contracts": len(df_all),
        "n_reference_contracts": int(df_all["is_ref"].sum()),
        "date_range": (df_all["date"].min(), df_all["date"].max()),
        "avg_contracts_per_date": len(df_all) / n_processed if n_processed > 0 else 0,
        "iv_stats": {
            "mean": float(df_all["implied_volatility"].mean()),
            "std": float(df_all["implied_volatility"].std()),
            "min": float(df_all["implied_volatility"].min()),
            "max": float(df_all["implied_volatility"].max()),
        },
    }

    print(f"\n  Results:")
    print(f"    Dates processed: {n_processed}")
    print(f"    Dates skipped: {n_skipped}")
    print(f"    Total contracts: {len(df_all):,}")
    print(f"    Avg contracts/date: {stats['avg_contracts_per_date']:.0f}")
    print(f"    Reference contracts: {stats['n_reference_contracts']}")
    print(f"    IV range: [{stats['iv_stats']['min']:.4f}, {stats['iv_stats']['max']:.4f}]")
    print(f"    IV mean: {stats['iv_stats']['mean']:.4f}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_pickle(str(output_path))
        print(f"    Saved to: {output_path}")

    return df_all, stats


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys

    data_dir = Path(".")
    output_dir = Path("./outputs/hyperiv")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 2 outputs
    print("Loading Stage 2 filtered data...")
    with open(data_dir / "outputs/stage2/filtered_dict.pkl", "rb") as f:
        filtered_dict = pickle.load(f)

    print("Loading market data...")
    with open(data_dir / "outputs/stage1/market_data.pkl", "rb") as f:
        market_data = pickle.load(f)

    fwd_df = market_data["fwd_df"]
    discount_df = market_data["discount_df"]

    # Prepare data
    df_all, stats = prepare_hyperiv_data(
        filtered_dict,
        fwd_df,
        discount_df,
        output_path=output_dir / "hyperiv_data.pkl",
    )

    print("\n  Sample data:")
    print(df_all.head(20).to_string())
