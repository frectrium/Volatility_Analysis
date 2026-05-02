"""Standard evaluation grid (12 strikes x 11 maturities) shared by all pipelines.

The grid is taken verbatim from Pipeline A (stage4_iv_sampling.py) so we keep
backward-compatibility with cached A-pipeline surfaces. Every pipeline must
produce a (12, 11) IV matrix on this grid for VolGAN consumption (Phase 2).

Phase-1 evaluation does NOT use this grid: each pipeline's continuous sigma(k, T)
model is queried at actual market quote points instead. The grid is only the
fixed representation that VolGAN trains on.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Strikes as K/F (forward moneyness). 12 levels.
MONEYNESS_STRIKES = np.array(
    [0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75]
)
# log-forward moneyness k = log(K/F)
LOG_MONEYNESS_GRID = np.log(MONEYNESS_STRIKES)

# Maturities in days, then years. 11 levels.
MATURITY_DAYS_GRID = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730])
MATURITY_YEARS_GRID = MATURITY_DAYS_GRID / 365.0

N_MONEYNESS = len(LOG_MONEYNESS_GRID)   # 12
N_MATURITY = len(MATURITY_YEARS_GRID)    # 11
GRID_SHAPE = (N_MONEYNESS, N_MATURITY)
N_GRID = N_MONEYNESS * N_MATURITY        # 132


def grid_kt():
    """Return (K, T) meshgrid points for surface evaluation.

    Returns:
        ks: (N_MONEYNESS, N_MATURITY) log-moneyness values.
        ts: (N_MONEYNESS, N_MATURITY) maturity in years.
    """
    ks, ts = np.meshgrid(LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID, indexing="ij")
    return ks, ts


def grid_flat():
    """Return flat (N_GRID, 2) array of (k, t) pairs."""
    ks, ts = grid_kt()
    return np.stack([ks.ravel(), ts.ravel()], axis=1)


def interp_surface_to_points(surface, k_query, t_query, *, fill_value=np.nan,
                             method="linear"):
    """Interpolate a (12, 11) standard-grid surface to arbitrary (k, t) points.

    Used in Phase 2: VolGAN outputs the grid surface and we evaluate it at
    actual market quote points. Returns NaN for query points outside the
    convex hull of the grid (no extrapolation by design).

    Args:
        surface: (12, 11) IV grid on (LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID).
        k_query: 1-D array of log-moneyness query values.
        t_query: 1-D array of maturity (years) query values.
        fill_value: returned for out-of-hull points. Default NaN (caller should
            mask before metrics).
        method: "linear" (default) or "nearest".

    Returns:
        sigma: 1-D array of interpolated IVs, same length as k_query.
    """
    surface = np.asarray(surface, dtype=np.float64)
    if surface.shape != GRID_SHAPE:
        raise ValueError(
            f"Surface shape {surface.shape} != expected {GRID_SHAPE}"
        )
    interp = RegularGridInterpolator(
        (LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID),
        surface,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )
    pts = np.stack([np.asarray(k_query, dtype=np.float64),
                    np.asarray(t_query, dtype=np.float64)], axis=1)
    return interp(pts)


def in_grid_hull(k_query, t_query):
    """Boolean mask: which query points fall inside the rectangular grid hull?

    Used in Phase 2 to drop test quotes outside the grid before RMSE.
    """
    k = np.asarray(k_query)
    t = np.asarray(t_query)
    return (
        (k >= LOG_MONEYNESS_GRID.min()) & (k <= LOG_MONEYNESS_GRID.max())
        & (t >= MATURITY_YEARS_GRID.min()) & (t <= MATURITY_YEARS_GRID.max())
    )
