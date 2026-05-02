"""
=============================================================================
CORRECTED PIPELINE — Part 1: Data Loading & Preprocessing
=============================================================================
Loads raw option data, interpolates forward/discount curves,
computes derived fields, and applies basic quality filters.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# ============================================================
# 1. LOAD RAW DATA
# ============================================================
def load_all_data(data_dir="."):
    """
    Load all CSV data files.
    Returns: dict with keys 'op_df', 'fwd_df', 'discount_df', 'St_df', 'list_exp', 'list_mny'
    """
    print("Loading data...")

    list_exp = pd.read_csv(f"{data_dir}/list_exp", index_col=0)
    list_exp.columns = ["Days"]

    list_mny = pd.read_csv(f"{data_dir}/list_mny", index_col=0)
    list_mny.columns = ["Moneyness"]

    St_df = pd.read_csv(f"{data_dir}/St_df", index_col=0, parse_dates=True)
    fwd_df = pd.read_csv(f"{data_dir}/fwd_df", index_col=0, parse_dates=True)
    discount_df = pd.read_csv(f"{data_dir}/discount_df", index_col=0, parse_dates=True)

    print("Loading op_df (this may take 1-2 minutes)...")
    op_df = pd.read_csv(f"{data_dir}/op_df", index_col=0, low_memory=False)
    op_df["date"] = pd.to_datetime(op_df["date"])
    print(f"Loaded op_df: {len(op_df):,} rows")

    return {
        "op_df": op_df,
        "fwd_df": fwd_df,
        "discount_df": discount_df,
        "St_df": St_df,
        "list_exp": list_exp,
        "list_mny": list_mny,
    }


# ============================================================
# 2. INTERPOLATE TERM STRUCTURE
# ============================================================
TENORS = np.array([0, 7, 14, 30, 60, 90, 180, 350])


def build_interp_map(grid_df, valid_dates, tenors=TENORS):
    """
    Build a dict: date -> interp1d function for a term-structure grid.
    Uses bounded extrapolation (clamp) instead of linear extrapolation
    to prevent unreliable values for long-dated options beyond the tenor grid.
    """
    grid_subset = grid_df.loc[grid_df.index.isin(valid_dates)]
    interp_map = {}
    for d, row in grid_subset.iterrows():
        vals = row.values.astype(float)
        # Use fill_value=(first, last) to CLAMP instead of extrapolate
        # This prevents nonsensical forward/discount for T >> 350 days
        interp_map[d] = interp1d(
            tenors, vals, kind="linear",
            bounds_error=False,
            fill_value=(vals[0], vals[-1]),  # Clamp to boundary values
        )
    return interp_map


def interpolate_column_vectorized(dates, maturities, interp_map):
    """
    Vectorized interpolation: for each row, look up the date's
    interpolation function and evaluate at the row's maturity.
    Returns a numpy array of interpolated values.
    """
    results = np.full(len(dates), np.nan)
    unique_dates = np.unique(dates)

    for d in unique_dates:
        if d in interp_map:
            mask = dates == d
            results[mask] = interp_map[d](maturities[mask])

    return results


# ============================================================
# 3. PREPROCESSING PIPELINE
# ============================================================
def preprocess(data, min_volume=10):
    """
    Full preprocessing pipeline:
    1. Filter to calls only
    2. Compute maturity in days
    3. Interpolate forward prices and discount factors
    4. Compute bid-ask spread, mid-price, relative spread
    5. Drop negative/zero spreads
    6. Apply volume & maturity filters
    7. Aggregate duplicates (date, exdate, strike)

    Returns: clean DataFrame ready for Moussa filtering
    """
    op_df = data["op_df"]
    fwd_df = data["fwd_df"]
    discount_df = data["discount_df"]

    # --- Step 1: Calls only ---
    df = op_df[op_df["cp_flag"] == "C"].copy()
    print(f"Calls only: {len(df):,} rows")

    # --- Step 2: Dates & Maturity ---
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    df["maturity_days"] = (df["exdate"] - df["date"]).dt.days

    # --- Step 3: Interpolate Forward & Discount ---
    valid_dates = df["date"].unique()
    fwd_map = build_interp_map(fwd_df, valid_dates)
    disc_map = build_interp_map(discount_df, valid_dates)

    dates_arr = df["date"].values
    mat_arr = df["maturity_days"].values.astype(float)

    df["fwd_price"] = interpolate_column_vectorized(dates_arr, mat_arr, fwd_map)
    df["discount"] = interpolate_column_vectorized(dates_arr, mat_arr, disc_map)

    # Drop rows where interpolation failed
    n_before = len(df)
    df.dropna(subset=["fwd_price", "discount"], inplace=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df):,} rows with NaN fwd/discount")

    # --- Step 4: Spread & Mid-Price ---
    df["spread"] = (df["best_offer"] - df["best_bid"]).abs()
    df["midP"] = (df["best_offer"] + df["best_bid"]) / 2.0
    df["spread_rel"] = df["spread"] / df["midP"].replace(0, np.nan)

    # --- Step 5: Drop negative/zero spread (offer <= bid) ---
    mask_bad_spread = df["best_offer"] <= df["best_bid"]
    n_bad = mask_bad_spread.sum()
    if n_bad > 0:
        print(f"  Dropping {n_bad} rows with offer <= bid")
        df = df[~mask_bad_spread].copy()

    # --- Step 6: Volume & Maturity Filter ---
    n_before = len(df)
    mask_liquid = (df["volume"] >= min_volume) & (df["maturity_days"] > 0)
    df = df[mask_liquid].copy()
    print(f"Volume filter (>= {min_volume}): {n_before:,} -> {len(df):,} rows")

    # --- Step 7: Aggregate duplicates ---
    # Group by unique contract identifier
    group_keys = ["date", "exdate", "strike"]
    g = df.groupby(group_keys)

    # Check if aggregation is actually needed
    if g.ngroups < len(df):
        agg_spec = {
            "volume": "sum",
            "spread": "mean",
            "best_bid": "mean",
            "best_offer": "mean",
            "fwd_price": "first",
            "discount": "first",
            "maturity_days": "first",
            "Spot": "first",
        }
        df_agg = g.agg(agg_spec).reset_index()
        # Volume-weighted mid-price
        df_agg["midP"] = g.apply(
            lambda x: (x["midP"] * x["volume"]).sum() / x["volume"].sum(),
            include_groups=False,
        ).values
        df_agg["spread_rel"] = df_agg["spread"] / df_agg["midP"]
        print(f"Aggregation: {len(df):,} -> {len(df_agg):,} rows")
        df = df_agg
    else:
        print("No duplicate contracts found — skipping aggregation.")
        df = df[
            ["date", "exdate", "strike", "volume", "spread", "best_bid",
             "best_offer", "fwd_price", "discount", "maturity_days", "Spot", "midP", "spread_rel"]
        ].copy()

    # --- Final: Build surface dictionary ---
    print("\nBuilding daily surface dictionary...")
    surface_dict = {}
    for date_val, daily_data in df.groupby("date"):
        sorted_surface = daily_data.sort_values(by=["exdate", "strike"]).reset_index(drop=True)
        surface_dict[date_val] = sorted_surface

    print(f"Surface dictionary: {len(surface_dict)} unique dates")
    print(f"Total rows across all dates: {len(df):,}")

    # Quick sanity stats
    print(f"\n[Data Summary]")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Maturity range: {df['maturity_days'].min()} to {df['maturity_days'].max()} days")
    print(f"  Strike range: {df['strike'].min():.0f} to {df['strike'].max():.0f}")
    print(f"  Median volume: {df['volume'].median():.0f}")
    print(f"  Median relative spread: {df['spread_rel'].median():.4f}")

    return surface_dict


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    data = load_all_data(".")
    surface_dict = preprocess(data, min_volume=10)

    # Peek at one date
    sample_date = sorted(surface_dict.keys())[5]
    print(f"\n[Sample: {sample_date.date()}]")
    print(surface_dict[sample_date][["exdate", "maturity_days", "strike", "midP", "volume"]].head(10))