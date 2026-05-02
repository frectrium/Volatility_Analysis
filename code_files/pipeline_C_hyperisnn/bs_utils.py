"""Black-Scholes utilities and Newton-Raphson IV inversion.

All NumPy, no autograd. Used for data preparation (computing IV from observed
prices) and for evaluation (inverting predicted prices back to IV for metrics).
"""

import math

import numpy as np
from scipy.special import erf as sp_erf


SQRT2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    return 0.5 * (1.0 + sp_erf(x / SQRT2))


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / SQRT_2PI


def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call price (no dividends)."""
    T_safe = np.maximum(T, 1e-12)
    sigT = sigma * np.sqrt(T_safe)
    d1 = (np.log(S / np.maximum(K, 1e-12)) + (r + 0.5 * sigma * sigma) * T) / np.maximum(sigT, 1e-12)
    d2 = d1 - sigT
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def bs_vega(S, K, T, r, sigma):
    """dC/dsigma."""
    T_safe = np.maximum(T, 1e-12)
    sigT = sigma * np.sqrt(T_safe)
    d1 = (np.log(S / np.maximum(K, 1e-12)) + (r + 0.5 * sigma * sigma) * T) / np.maximum(sigT, 1e-12)
    return S * np.sqrt(T_safe) * norm_pdf(d1)


def implied_vol(price, S, K, T, r, n_iter=25):
    """Vectorized Newton-Raphson IV inversion. Returns NaN where no convergence."""
    price = np.asarray(price, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    # Manaster-Koehler initial guess
    sigma = np.sqrt(2 * np.pi / np.maximum(T, 1e-6)) * (price / S)
    sigma = np.clip(sigma, 0.05, 2.0)
    for _ in range(n_iter):
        bs = bs_call(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        sigma = sigma - (bs - price) / np.maximum(v, 1e-8)
        sigma = np.clip(sigma, 1e-3, 3.0)
    bs = bs_call(S, K, T, r, sigma)
    bad = np.abs(bs - price) > max(0.5, 0.01 * S)
    sigma = sigma.copy()
    sigma[bad] = np.nan
    return sigma
