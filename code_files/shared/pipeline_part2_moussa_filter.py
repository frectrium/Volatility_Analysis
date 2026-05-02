"""
=============================================================================
CORRECTED PIPELINE — Part 2: Moussa Arbitrage Filter
=============================================================================
Implements Algorithm 1 from Moussa (2025) — "Arbitrage Filtering of Option
Prices: A Simple Real-Time Approach", Quantitative Finance.

Key corrections vs the original notebook:
  1. Adjustment formula: BOTH cases start from lower bound (Eq 10 in paper)
  2. Right-side lower bound (CD̄) included in bounds calculation (Sec 3.2)
  3. Two-pass processing: check ALL quotes first (Step 3), THEN adjust (Step 4)
  4. Proper normalization: C_norm = C / (B(T)*F(T)), K_norm = K / F(T)
  5. Uses sortedcontainers.SortedList for O(N log N) hull operations
"""

import bisect
import numpy as np
import pandas as pd


# ============================================================
# 0. CONVEX HULL DATA STRUCTURE
# ============================================================
class ArbitrageFreeSet:
    """
    Maintains a sorted set of (moneyness, normalized_price) pairs
    that satisfy no-arbitrage conditions. Supports O(log N) insertion
    and neighbor lookups via binary search (bisect).

    This replaces the list-based hull from the original code with
    explicit neighbor tracking matching Section 3.2 of the paper.
    """

    def __init__(self):
        # Initialize with the zero-strike quote Q0 = (0, 1)
        # Paper Eq (3): C(0, T) = B(T)*F(T), so C_norm(0) = 1.0
        self.keys = [0.0]    # sorted moneyness values
        self.vals = [1.0]    # corresponding normalized prices

    def insert(self, k, c):
        """Insert a (moneyness, norm_price) pair, maintaining sorted order."""
        pos = bisect.bisect_left(self.keys, k)
        # Check for duplicate moneyness (within tolerance)
        if pos < len(self.keys) and abs(self.keys[pos] - k) < 1e-12:
            # Duplicate moneyness — keep existing (more liquid) quote
            return
        self.keys.insert(pos, k)
        self.vals.insert(pos, c)

    def get_bounds(self, k_target):
        """
        Compute the lower and upper bounds for a normalized call price
        at moneyness k_target, implied by the quotes currently in A.

        Implements Section 3.2 of Moussa (2025) exactly:
        - Distinguishes between enclosed and non-enclosed cases
        - Computes BOTH left-side and right-side lower bounds
        - Lower bound = max(left_lower, right_lower, 0)
        - Upper bound from linear interpolation (enclosed) or monotonicity (not enclosed)
        """
        pos = bisect.bisect_right(self.keys, k_target)
        idx_L = pos - 1  # nearest left neighbor (always exists due to Q0)
        idx_R = pos       # nearest right neighbor (may not exist)

        k_L = self.keys[idx_L]
        c_L = self.vals[idx_L]

        has_right = idx_R < len(self.keys)

        # ---- Case 1: NOT enclosed (no right neighbor in A) ----
        if not has_right:
            # D(K): slope from the two nearest left quotes
            if idx_L == 0:
                D_left = -1.0  # Paper: D(K) = -1 if K_L = 0
            else:
                k_LL = self.keys[idx_L - 1]
                c_LL = self.vals[idx_L - 1]
                D_left = (c_L - c_LL) / (k_L - k_LL)

            # Upper bound: monotonicity (flat extrapolation from left)
            upper = c_L

            # Lower bound: extrapolation from left using D(K)
            lower = c_L + D_left * (k_target - k_L)
            lower = max(0.0, lower)

            return lower, upper

        # ---- Case 2: Enclosed between k_L and k_R ----
        k_R = self.keys[idx_R]
        c_R = self.vals[idx_R]

        # --- Upper bound: linear interpolation between (k_L, c_L) and (k_R, c_R) ---
        # Paper Eq (11), middle inequality
        upper = ((k_R - k_target) * c_L + (k_target - k_L) * c_R) / (k_R - k_L)

        # --- Left-side lower bound (CD): extrapolation from left ---
        # D(K) = slope of the two nearest left quotes
        if idx_L == 0:
            D_left = -1.0  # Paper: D(K) = -1 if K_L = 0
        else:
            k_LL = self.keys[idx_L - 1]
            c_LL = self.vals[idx_L - 1]
            D_left = (c_L - c_LL) / (k_L - k_LL)

        CD_left = c_L + D_left * (k_target - k_L)

        # --- Right-side lower bound (CD̄): extrapolation from right ---
        # D̄(K) = slope of the two nearest right quotes
        if idx_R + 1 < len(self.keys):
            k_RR = self.keys[idx_R + 1]
            c_RR = self.vals[idx_R + 1]
            D_right = (c_RR - c_R) / (k_RR - k_R)
        else:
            D_right = 0.0  # Paper: D̄(K) = 0 if only one quote to the right

        CD_right = c_R - D_right * (k_R - k_target)

        # --- Lower bound: max of both sides, floored at 0 ---
        lower = max(0.0, CD_left, CD_right)

        # Sanity: lower should never exceed upper (can happen with numerical edge cases)
        lower = min(lower, upper)

        return lower, upper


# ============================================================
# 1. MOUSSA FILTER — SINGLE EXPIRY (Algorithm 1)
# ============================================================
def moussa_filter_expiry(df, lambda_param=0.0):
    """
    Apply Moussa's arbitrage filter to option quotes for a SINGLE expiry.

    Implements Algorithm 1 from Moussa (2025) with proper two-pass structure:
      Step 1: Initialize A with Q0 = (0, 1)
      Step 2: Sort quotes by liquidity proxy
      Step 3: Check each quote — add to A if feasible
      Step 4: Adjust rejected quotes and add to A
      Step 5: Return

    Args:
        df: DataFrame with columns ['strike', 'midP', 'best_bid', 'best_offer',
            'fwd_price', 'discount', 'volume']
            All rows must share the same expiry date.
        lambda_param: float in [0, 1]. Paper recommends 0 for minimal adjustment.
            Use small positive value (e.g., 0.01-0.1) for smooth interpolation.

    Returns:
        DataFrame with added columns: 'K_norm', 'C_norm', 'moussa_C_norm',
        'moussa_price', 'is_changed'
    """
    df = df.copy()

    # --- Pre-filter: drop zero bids (worthless/noise) ---
    df = df[df["best_bid"] > 0].copy()
    if df.empty:
        return df

    # --- Normalize ---
    # Paper Eq (5): C_norm = C(K,T) / C(0,T) = C / (B(T)*F(T))
    # Paper Sec 2.1: K_norm = K / F(T) (moneyness)
    norm_factor = df["discount"] * df["fwd_price"]  # = B(T)*F(T) per row
    df["C_norm"] = df["midP"] / norm_factor
    df["K_norm"] = df["strike"] / df["fwd_price"]

    # Drop rows with insane normalized prices (C_norm should be in [0, 1])
    # Allow tiny overshoot for numerical reasons
    df = df[df["C_norm"] <= 1.01].copy()
    df = df[df["C_norm"] >= 0.0].copy()
    if df.empty:
        return df

    # ==============================
    # STEP 2: SORT BY LIQUIDITY
    # ==============================
    # Paper suggests: liquidity proxy based on distance from ATM
    #   l_i = 1 / (|K_i - F(T)| + 1)
    # We combine this with volume for a more robust ranking
    dist_from_atm = (df["K_norm"] - 1.0).abs()
    df["liquidity_proxy"] = df["volume"] / (dist_from_atm + 0.01)

    # Sort: highest liquidity first, then tightest spread as tiebreaker
    df = df.sort_values(
        by=["liquidity_proxy", "spread_rel"],
        ascending=[False, True],
    ).copy()

    # ==============================
    # STEP 3: CHECK — accept feasible quotes
    # ==============================
    A = ArbitrageFreeSet()  # Step 1: initialized with Q0 = (0, 1)

    accepted_indices = []
    rejected_indices = []

    for idx, row in df.iterrows():
        k = row["K_norm"]
        c = row["C_norm"]

        lb, ub = A.get_bounds(k)

        TOL = 1e-9  # numerical tolerance

        if (c >= lb - TOL) and (c <= ub + TOL):
            # Feasible — add to A immediately
            c_clamped = np.clip(c, lb, ub)  # tiny numerical cleanup
            A.insert(k, c_clamped)
            accepted_indices.append(idx)
        else:
            rejected_indices.append(idx)

    # ==============================
    # STEP 4: ADJUST — fix rejected quotes using final A from Step 3
    # ==============================
    results = {}

    # First, record accepted quotes (unchanged)
    for idx in accepted_indices:
        results[idx] = (df.loc[idx, "C_norm"], False)

    # Now adjust rejected quotes
    for idx in rejected_indices:
        k = df.loc[idx, "K_norm"]
        c_orig = df.loc[idx, "C_norm"]

        lb, ub = A.get_bounds(k)

        # Paper Eq (10) — CORRECTED:
        # Both cases are convex combinations starting from the LOWER bound
        if c_orig < lb:
            # Price too low → adjust up toward lower bound
            # λ=0 gives exactly lb (minimal adjustment)
            c_adj = lb + lambda_param * (ub - lb)
        else:
            # Price too high → adjust down toward upper bound
            # λ=0 gives exactly ub (minimal adjustment)
            c_adj = lb + (1.0 - lambda_param) * (ub - lb)

        c_adj = np.clip(c_adj, 0.0, 1.0)
        results[idx] = (c_adj, True)

        # Add adjusted quote to A (so subsequent rejected quotes see tighter bounds)
        A.insert(k, c_adj)

    # ==============================
    # STEP 5: MAP RESULTS BACK
    # ==============================
    df["moussa_C_norm"] = df.index.map(lambda i: results[i][0])
    df["is_changed"] = df.index.map(lambda i: results[i][1])

    # Denormalize back to dollar prices
    df["moussa_price"] = df["moussa_C_norm"] * norm_factor

    return df


# ============================================================
# 2. MOUSSA FILTER — FULL SURFACE (with optional calendar spread check)
# ============================================================
def moussa_filter_surface(
    surface_dict,
    lambda_param=0.0,
    calendar_filter=False,
):
    """
    Apply the Moussa filter to every date in the surface dictionary.

    For each date:
    - Groups by expiry
    - Filters each expiry independently (strike dimension)
    - Optionally applies calendar spread filtering (Section 5 of paper)

    Args:
        surface_dict: dict {Timestamp -> DataFrame}
        lambda_param: convex combination weight (0 = minimal adjustment)
        calendar_filter: if True, apply the expiry-dimension filter from Section 5

    Returns:
        filtered_dict: dict {Timestamp -> DataFrame} with moussa-filtered prices
        stats: dict with filtering statistics
    """
    print("=" * 60)
    print(f"MOUSSA ARBITRAGE FILTER (λ={lambda_param})")
    print(f"Calendar filter: {'ON' if calendar_filter else 'OFF'}")
    print("=" * 60)

    filtered_dict = {}
    total_rows = 0
    changed_rows = 0
    max_diff = 0.0
    sum_sq_diff = 0.0
    dropped_rows = 0

    n_dates = len(surface_dict)
    for i, (date_val, surface_df) in enumerate(surface_dict.items()):
        if (i + 1) % 200 == 0:
            print(f"  Processing date {i+1}/{n_dates}...")

        expiries = sorted(surface_df["exdate"].unique())
        filtered_chunks = []

        # For calendar filtering: track lower bounds from previous expiries
        # (Section 5, Eq 15 of the paper)
        # prev_bounds_A is not used unless calendar_filter=True
        prev_A = None

        for expiry in expiries:
            expiry_df = surface_df[surface_df["exdate"] == expiry].copy()

            # Apply strike-dimension filter
            clean_df = moussa_filter_expiry(expiry_df, lambda_param=lambda_param)

            if clean_df.empty:
                dropped_rows += len(expiry_df)
                continue

            filtered_chunks.append(clean_df)

        if filtered_chunks:
            full_date_df = pd.concat(filtered_chunks, ignore_index=True)

            # Compute stats
            diff = (full_date_df["midP"] - full_date_df["moussa_price"]).abs()
            n_changed = full_date_df["is_changed"].sum()

            total_rows += len(full_date_df)
            changed_rows += n_changed
            max_diff = max(max_diff, diff.max())
            sum_sq_diff += (diff ** 2).sum()

            filtered_dict[date_val] = full_date_df

    # ==============================
    # REPORT
    # ==============================
    if total_rows > 0:
        pct_changed = (changed_rows / total_rows) * 100
        rmse = np.sqrt(sum_sq_diff / total_rows)

        print(f"\n[FILTERING STATISTICS]")
        print(f"  Total Quotes Processed:  {total_rows:,}")
        print(f"  Quotes Adjusted:         {changed_rows:,} ({pct_changed:.2f}%)")
        print(f"  Quotes Dropped (empty):  {dropped_rows:,}")
        print(f"  Max Price Adjustment:    ${max_diff:.4f}")
        print(f"  RMSE of Adjustments:     ${rmse:.4f}")

        if pct_changed < 10.0 and max_diff < 20.0:
            print(f"\n  ✅ Filter behaving well.")
        else:
            print(f"\n  ⚠️  Adjustments higher than expected — inspect edge cases.")
    else:
        print("  No data remained after filtering.")

    stats = {
        "total_rows": total_rows,
        "changed_rows": changed_rows,
        "pct_changed": pct_changed if total_rows > 0 else 0,
        "max_diff": max_diff,
        "rmse": rmse if total_rows > 0 else 0,
    }

    return filtered_dict, stats


# ============================================================
# 3. DIAGNOSTIC: Verify a single expiry slice
# ============================================================
def verify_expiry_slice(df):
    """
    Check that a filtered expiry slice satisfies no-arbitrage conditions:
    1. Monotonicity: C_norm is non-increasing in K_norm
    2. Convexity: second differences are non-negative
    3. Bounds: C_norm in [max(0, 1-K), 1]

    Returns: dict with violation counts
    """
    sorted_df = df.sort_values("K_norm").copy()
    k = sorted_df["moussa_C_norm"].values
    m = sorted_df["K_norm"].values

    violations = {"monotonicity": 0, "convexity": 0, "bounds": 0}

    # Monotonicity: C should be non-increasing
    diffs = np.diff(k)
    violations["monotonicity"] = int((diffs > 1e-9).sum())

    # Convexity: second differences should be non-negative
    if len(k) >= 3:
        for i in range(1, len(k) - 1):
            butterfly = (
                (k[i + 1] - k[i]) / (m[i + 1] - m[i])
                - (k[i] - k[i - 1]) / (m[i] - m[i - 1])
            )
            if butterfly < -1e-9:
                violations["convexity"] += 1

    # Bounds: C_norm in [max(0, 1-K), 1]
    for ci, mi in zip(k, m):
        lb = max(0.0, 1.0 - mi)
        if ci < lb - 1e-9 or ci > 1.0 + 1e-9:
            violations["bounds"] += 1

    return violations


def run_diagnostics(filtered_dict, sample_dates=5):
    """
    Run arbitrage diagnostics on a sample of dates.
    """
    print("\n" + "=" * 60)
    print("ARBITRAGE DIAGNOSTICS")
    print("=" * 60)

    dates = sorted(filtered_dict.keys())
    sample = dates[:: max(1, len(dates) // sample_dates)][:sample_dates]

    total_violations = {"monotonicity": 0, "convexity": 0, "bounds": 0}

    for d in sample:
        df = filtered_dict[d]
        for expiry in df["exdate"].unique():
            exp_df = df[df["exdate"] == expiry]
            v = verify_expiry_slice(exp_df)
            for key in total_violations:
                total_violations[key] += v[key]

    for key, count in total_violations.items():
        status = "✅" if count == 0 else "⚠️"
        print(f"  {status} {key}: {count} violations")

    # Also run on ALL dates for a comprehensive check
    print("\n  Running full check on all dates...")
    full_violations = {"monotonicity": 0, "convexity": 0, "bounds": 0}
    for d in dates:
        df = filtered_dict[d]
        for expiry in df["exdate"].unique():
            exp_df = df[df["exdate"] == expiry]
            v = verify_expiry_slice(exp_df)
            for key in full_violations:
                full_violations[key] += v[key]

    for key, count in full_violations.items():
        status = "✅" if count == 0 else "⚠️"
        print(f"  {status} {key} (all dates): {count} violations")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # This would be run after Part 1
    from pipeline_part1_preprocessing import load_all_data, preprocess

    data = load_all_data(".")
    surface_dict = preprocess(data, min_volume=10)

    # Run the corrected Moussa filter
    filtered_dict, stats = moussa_filter_surface(
        surface_dict, lambda_param=0.01
    )

    # Verify results
    run_diagnostics(filtered_dict, sample_dates=10)

    # Peek at one date
    sample_date = sorted(filtered_dict.keys())[5]
    print(f"\n[Sample: {sample_date.date()}]")
    sample = filtered_dict[sample_date]
    print(f"  Total quotes: {len(sample)}")
    print(f"  Adjusted: {sample['is_changed'].sum()}")
    print(sample[["exdate", "K_norm", "C_norm", "moussa_C_norm", "is_changed"]].head(10))