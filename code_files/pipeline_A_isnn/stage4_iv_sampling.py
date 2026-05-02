"""
=============================================================================
PIPELINE — Part 4: Surface Sampling & IV Inversion
=============================================================================
Evaluates each day's trained ISNN model on a fixed (moneyness, maturity) grid,
then inverts Black-Scholes to obtain implied volatilities.

Grid specification (matching Zhang et al. 2021):
  - 14 log-forward moneyness points
  - 11 maturity tenors
  - Total: 154 grid points per day

Output: matrix of shape (n_dates, 154) with implied volatilities.
"""

import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import brentq


# ============================================================
# 0. GRID DEFINITION (Zhang et al. 2021)
# ============================================================
# Removed extreme strikes K/F=0.6 and K/F=2.0 — they are deep OTM with
# near-zero time value, causing IV inversion failures and noisy extrapolation.
# The remaining 12 strikes still span a wide range (0.8 to 1.75).
MONEYNESS_STRIKES = np.array([
    0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75
])
MONEYNESS_GRID = np.log(MONEYNESS_STRIKES)  # log-forward moneyness m = log(K/F)

MATURITY_DAYS_GRID = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730])
MATURITY_GRID = MATURITY_DAYS_GRID / 365.0  # in years

N_MONEYNESS = len(MONEYNESS_GRID)   # 14
N_MATURITY = len(MATURITY_GRID)      # 11
N_GRID = N_MONEYNESS * N_MATURITY    # 154


# ============================================================
# 1. BLACK-SCHOLES NORMALIZED FORMULAS
# ============================================================
def bs_normalized_call(m, T, sigma):
    """
    Normalized Black-Scholes call price: C_norm = N(d1) - exp(m)*N(d2).

    This equals C / (B(T)*F(T)) where C is the dollar call price,
    B(T) is the discount factor, and F(T) is the forward price.

    Args:
        m: log-forward moneyness log(K/F), shape (N,)
        T: time to maturity in years, shape (N,)
        sigma: implied volatility, shape (N,)

    Returns:
        C_norm: normalized call price, shape (N,)
    """
    sqrt_T = np.sqrt(T)
    d1 = (-m + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return norm.cdf(d1) - np.exp(m) * norm.cdf(d2)


def bs_vega_normalized(m, T, sigma):
    """
    Vega of normalized BS call price w.r.t. sigma.

    dC_norm/dsigma = N'(d1) * sqrt(T)

    where N'(x) = (1/sqrt(2*pi)) * exp(-x^2/2).

    Args:
        m: log-forward moneyness, shape (N,)
        T: time to maturity in years, shape (N,)
        sigma: implied volatility, shape (N,)

    Returns:
        vega: shape (N,)
    """
    sqrt_T = np.sqrt(T)
    d1 = (-m + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    return norm.pdf(d1) * sqrt_T


# ============================================================
# 2. IMPLIED VOLATILITY INVERSION
# ============================================================
def implied_vol_newton(C_norm_target, m, T, sigma_init=0.3, max_iter=50, tol=1e-8):
    """
    Vectorized Newton-Raphson implied volatility inversion.

    Solves: bs_normalized_call(m, T, sigma) = C_norm_target

    Args:
        C_norm_target: target normalized call prices, shape (N,)
        m: log-forward moneyness, shape (N,)
        T: time to maturity in years, shape (N,)
        sigma_init: initial guess for IV
        max_iter: maximum Newton iterations
        tol: convergence tolerance

    Returns:
        sigma: implied volatilities, shape (N,). NaN for non-converged points.
    """
    sigma = np.full_like(C_norm_target, sigma_init, dtype=np.float64)
    active = np.ones(len(sigma), dtype=bool)

    for _ in range(max_iter):
        if not active.any():
            break

        c_model = bs_normalized_call(m[active], T[active], sigma[active])
        vega = bs_vega_normalized(m[active], T[active], sigma[active])

        diff = c_model - C_norm_target[active]

        # Avoid division by zero vega
        safe_vega = np.where(np.abs(vega) > 1e-12, vega, 1e-12)
        update = diff / safe_vega

        # Damped update to avoid overshooting
        update = np.clip(update, -0.5, 0.5)
        sigma[active] -= update

        # Clamp sigma to reasonable range
        sigma[active] = np.clip(sigma[active], 1e-4, 5.0)

        # Check convergence
        converged_mask = np.abs(diff) < tol
        active_indices = np.where(active)[0]
        active[active_indices[converged_mask]] = False

    # Mark non-converged as NaN
    sigma[active] = np.nan
    return sigma


def implied_vol_brentq_fallback(C_norm_target, m, T, sigma_lo=0.001, sigma_hi=5.0):
    """
    Scalar Brentq fallback for individual non-converged points.

    Args:
        C_norm_target: single target normalized call price
        m: single log-forward moneyness
        T: single time to maturity in years

    Returns:
        sigma: implied volatility (float), or NaN if brentq fails
    """
    def objective(sigma):
        return bs_normalized_call(np.array([m]), np.array([T]), np.array([sigma]))[0] - C_norm_target

    try:
        # Check that the root is bracketed
        f_lo = objective(sigma_lo)
        f_hi = objective(sigma_hi)
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(objective, sigma_lo, sigma_hi, xtol=1e-10, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


# ============================================================
# 3. SURFACE SAMPLING
# ============================================================
def sample_surface_single_date(model, moneyness_grid=MONEYNESS_GRID, maturity_grid=MATURITY_GRID):
    """
    Evaluate ISNN on the full grid and invert to get IV surface.

    The ISNN takes (K/F, T_years) and outputs C_norm = C/(B*F).
    We convert from log-moneyness grid to K/F = exp(m) for the ISNN input,
    then invert BS to get implied volatility.

    Args:
        model: trained ISNN2_OptionSurface
        moneyness_grid: log-forward moneyness values, shape (n_m,)
        maturity_grid: maturity values in years, shape (n_t,)

    Returns:
        iv_surface: shape (n_m * n_t,) — flattened row-major (moneyness varies first)
        n_nan: number of NaN values in the result
    """
    # Build meshgrid: moneyness x maturity
    M, T = np.meshgrid(moneyness_grid, maturity_grid, indexing="ij")
    m_flat = M.ravel().astype(np.float64)   # log-moneyness for BS inversion
    T_flat = T.ravel().astype(np.float64)   # maturity in years

    # ISNN input: K/F = exp(m), T_years
    kf_flat = np.exp(m_flat)

    x0 = torch.tensor(kf_flat, dtype=torch.float32).unsqueeze(1)
    t0 = torch.tensor(T_flat, dtype=torch.float32).unsqueeze(1)

    # Evaluate ISNN
    model.eval()
    with torch.no_grad():
        C_norm = model(x0, t0).squeeze(1).numpy().astype(np.float64)

    # Clamp C_norm to valid BS range:
    #   Intrinsic value: max(0, 1 - K/F) = max(0, 1 - exp(m))
    #   Upper bound: 1 (for a call)
    intrinsic = np.maximum(0.0, 1.0 - np.exp(m_flat))
    C_norm_clamped = np.clip(C_norm, intrinsic + 1e-8, 1.0 - 1e-8)

    # Skip points where the price is essentially at intrinsic (no time value)
    has_time_value = (C_norm_clamped - intrinsic) > 1e-7

    # Newton-Raphson inversion
    iv = np.full_like(C_norm_clamped, np.nan)
    if has_time_value.any():
        iv[has_time_value] = implied_vol_newton(
            C_norm_clamped[has_time_value],
            m_flat[has_time_value],
            T_flat[has_time_value],
        )

    # Brentq fallback for NaN values that had time value
    nan_mask = np.isnan(iv) & has_time_value
    n_fallback = nan_mask.sum()
    if n_fallback > 0:
        for idx in np.where(nan_mask)[0]:
            iv[idx] = implied_vol_brentq_fallback(
                C_norm_clamped[idx], m_flat[idx], T_flat[idx]
            )

    # Interpolate remaining NaN values on the grid
    n_nan = np.isnan(iv).sum()
    if n_nan > 0 and n_nan < len(iv):
        iv = _interpolate_nan_on_grid(iv, len(moneyness_grid), len(maturity_grid))

    n_nan_final = np.isnan(iv).sum()
    return iv, n_nan_final


def _interpolate_nan_on_grid(iv_flat, n_m, n_t):
    """
    Interpolate NaN values using nearest valid neighbors on the (moneyness, maturity) grid.
    Simple approach: for each NaN, take the mean of its non-NaN neighbors.
    Falls back to column/row mean if no neighbors available.
    """
    iv_grid = iv_flat.reshape(n_m, n_t).copy()

    # Iterative filling: expand from valid neighbors
    for _pass in range(max(n_m, n_t)):
        nan_mask = np.isnan(iv_grid)
        if not nan_mask.any():
            break

        filled = iv_grid.copy()
        for i in range(n_m):
            for j in range(n_t):
                if not nan_mask[i, j]:
                    continue
                neighbors = []
                if i > 0 and not np.isnan(iv_grid[i - 1, j]):
                    neighbors.append(iv_grid[i - 1, j])
                if i < n_m - 1 and not np.isnan(iv_grid[i + 1, j]):
                    neighbors.append(iv_grid[i + 1, j])
                if j > 0 and not np.isnan(iv_grid[i, j - 1]):
                    neighbors.append(iv_grid[i, j - 1])
                if j < n_t - 1 and not np.isnan(iv_grid[i, j + 1]):
                    neighbors.append(iv_grid[i, j + 1])
                if neighbors:
                    filled[i, j] = np.mean(neighbors)
        iv_grid = filled

    # If still NaN, fill with global mean
    global_mean = np.nanmean(iv_grid)
    if np.isnan(global_mean):
        global_mean = 0.2  # fallback
    iv_grid = np.where(np.isnan(iv_grid), global_mean, iv_grid)

    return iv_grid.ravel()


# ============================================================
# 4. BATCH SAMPLING (all dates)
# ============================================================
def sample_all_surfaces(models_dict, moneyness_grid=MONEYNESS_GRID, maturity_grid=MATURITY_GRID):
    """
    Sample IV surfaces for all dates from trained ISNN models.

    Args:
        models_dict: dict {date -> (model, processor, losses)} from ISNN fitting
        moneyness_grid: log-forward moneyness grid
        maturity_grid: maturity grid in years

    Returns:
        iv_matrix: np.ndarray shape (n_dates, 154)
        dates: sorted list of Timestamps
    """
    print("\n" + "=" * 60)
    print("SURFACE SAMPLING & IV INVERSION")
    print("=" * 60)

    n_grid = len(moneyness_grid) * len(maturity_grid)
    dates = sorted(models_dict.keys())
    n_dates = len(dates)

    iv_matrix = np.zeros((n_dates, n_grid))
    total_nan = 0

    for i, d in enumerate(dates):
        model, processor, losses = models_dict[d]
        iv_surface, n_nan = sample_surface_single_date(model, moneyness_grid, maturity_grid)
        iv_matrix[i] = iv_surface
        total_nan += n_nan

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{n_dates}] {d.date()} | NaN count: {n_nan}")

    # Summary statistics
    print(f"\n[SAMPLING SUMMARY]")
    print(f"  Dates sampled: {n_dates}")
    print(f"  Grid size: {n_grid} points per date")
    print(f"  Total NaN after interpolation: {total_nan}")
    print(f"  IV range: [{np.nanmin(iv_matrix):.4f}, {np.nanmax(iv_matrix):.4f}]")
    print(f"  IV mean: {np.nanmean(iv_matrix):.4f}")
    print(f"  IV std:  {np.nanstd(iv_matrix):.4f}")

    # Warn if IV values look unreasonable
    n_extreme = np.sum((iv_matrix < 0.01) | (iv_matrix > 3.0))
    if n_extreme > 0:
        pct = 100 * n_extreme / iv_matrix.size
        print(f"  WARNING: {n_extreme} points ({pct:.2f}%) with extreme IV (<0.01 or >3.0)")

    # Clip extreme IV values to prevent downstream issues (VAE training corruption)
    # SPX IV rarely exceeds 150% even in crisis; deep OTM extrapolation is unreliable
    IV_CLIP_UPPER = 1.5
    IV_CLIP_LOWER = 0.01
    n_clipped = np.sum((iv_matrix > IV_CLIP_UPPER) | (iv_matrix < IV_CLIP_LOWER))
    iv_matrix = np.clip(iv_matrix, IV_CLIP_LOWER, IV_CLIP_UPPER)
    if n_clipped > 0:
        pct = 100 * n_clipped / iv_matrix.size
        print(f"  Clipped {n_clipped} points ({pct:.2f}%) to [{IV_CLIP_LOWER}, {IV_CLIP_UPPER}]")

    return iv_matrix, dates


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Stage 4 requires trained ISNN models from Stage 3.")
    print("Invoked via shared_eval.adapters.pipeline_a_isnn (the unified runner).")
