"""
=============================================================================
Data Utilities for HyperIV Training
=============================================================================
Provides:
  - OptionDataset: PyTorch Dataset for loading option data per date
  - Black-Scholes pricing, delta, vega, implied volatility helpers
  - SSVI parameterisation
  - Reference set selection utilities
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.stats import norm
from scipy.optimize import brentq


# ============================================================
# 1. BLACK-SCHOLES FUNCTIONS
# ============================================================
def black_scholes_price(K, t, F, is_call, sigma, r=0.0):
    """
    Black-Scholes price for European options.

    Args:
        K: strike price(s)
        t: time to maturity in years
        F: forward price(s)
        is_call: 1 for call, -1 for put
        sigma: implied volatility
        r: risk-free rate

    Returns:
        option price(s)
    """
    K, t, F, sigma = np.asarray(K, float), np.asarray(t, float), \
                      np.asarray(F, float), np.asarray(sigma, float)
    is_call = np.asarray(is_call, float)

    k = np.log(K / F)
    d1 = (-k + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    discount = np.exp(-r * t) if np.any(r != 0) else 1.0

    call_price = discount * F * (norm.cdf(d1) - np.exp(k) * norm.cdf(d2))
    put_price = discount * F * (np.exp(k) * norm.cdf(-d2) - norm.cdf(-d1))

    price = np.where(is_call == 1, call_price, put_price)
    return price


def black_scholes_delta(K, t, F, is_call, sigma):
    """Black-Scholes delta."""
    K, t, F, sigma = np.asarray(K, float), np.asarray(t, float), \
                      np.asarray(F, float), np.asarray(sigma, float)
    is_call = np.asarray(is_call, float)

    k = np.log(K / F)
    d1 = (-k + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))

    delta = np.where(is_call == 1, norm.cdf(d1), norm.cdf(d1) - 1)
    return delta


def black_scholes_vega(K, t, F, sigma, r=0.0):
    """Black-Scholes vega."""
    K, t, F, sigma = np.asarray(K, float), np.asarray(t, float), \
                      np.asarray(F, float), np.asarray(sigma, float)

    k = np.log(K / F)
    d1 = (-k + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    discount = np.exp(-r * t) if np.any(r != 0) else 1.0

    return discount * F * norm.pdf(d1) * np.sqrt(t)


def calc_implied_volatility(row):
    """
    Calculate implied volatility from a DataFrame row using Brent's method.

    Expected columns: strike_price, tau, forward_price, is_call,
                      option_price, risk_free_rate
    """
    K = row["strike_price"]
    t = row["tau"]
    F = row["forward_price"]
    is_call = row["is_call"]
    V = row["option_price"]
    r = row.get("risk_free_rate", 0.0)

    def objective(sigma):
        return black_scholes_price(K, t, F, is_call, sigma, r) - V

    try:
        iv = brentq(objective, 1e-4, 5.0, xtol=1e-10, maxiter=200)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ============================================================
# 2. SSVI PARAMETERISATION
# ============================================================
def SSVI(k, t, params):
    """
    Surface SVI (SSVI) total implied variance.

    w(k, t) = (sigma^2 * t / 2) * (1 + rho * eta * (zeta^2 * t)^(-gamma) * k
               + sqrt((eta * (zeta^2 * t)^(-gamma) * k + rho)^2 + 1 - rho^2))

    Args:
        k: log-moneyness
        t: time to maturity
        params: (sigma, gamma, eta, rho)

    Returns:
        w: total implied variance
    """
    sigma, gamma, eta, rho = params
    k, t = np.asarray(k, float), np.asarray(t, float)

    zeta2t = (sigma**2 * t)
    phi = eta * zeta2t**(-gamma)

    w = (zeta2t / 2) * (1 + rho * phi * k +
         np.sqrt((phi * k + rho)**2 + 1 - rho**2))
    return w


# ============================================================
# 3. REFERENCE SET SELECTION
# ============================================================
def find_closest_elements(df, column, targets, include_groups=True):
    """
    For a DataFrame grouped by some key, find the rows closest to
    each target value in the given column.

    Args:
        df: DataFrame (may be a group from groupby)
        column: column name to match against
        targets: list of target values

    Returns:
        DataFrame with one row per target
    """
    results = []
    for target in targets:
        idx = (df[column] - target).abs().idxmin()
        results.append(df.loc[idx])
    return pd.DataFrame(results)


def find_closest_option(df, ttm, delta):
    """
    Find the option closest to a target (ttm, delta) pair.

    Args:
        df: DataFrame with columns 'time_to_maturity' and 'delta'
        ttm: target time to maturity in days
        delta: target delta value

    Returns:
        dict with the closest option's data
    """
    # First filter to closest TTM
    ttm_diff = (df["time_to_maturity"] - ttm).abs()
    closest_ttm = ttm_diff.min()
    ttm_candidates = df[ttm_diff == closest_ttm]

    # Then find closest delta
    delta_diff = (ttm_candidates["delta"] - delta).abs()
    best_idx = delta_diff.idxmin()

    return df.loc[best_idx].to_dict()


def find_optimal_k(target_delta, tau, forward_price, is_call, ssvi_params):
    """
    Find the log-moneyness k that achieves a target delta under SSVI.

    Uses Brent's method to solve: BS_delta(K=F*exp(k), tau, F, is_call,
    sigma=sqrt(SSVI(k,tau)/tau)) = target_delta.
    """
    def objective(k):
        w = SSVI(k, tau, ssvi_params)
        sigma = np.sqrt(max(w / tau, 1e-10))
        K = forward_price * np.exp(k)
        d = black_scholes_delta(K, tau, forward_price, is_call, sigma)
        return d - target_delta

    try:
        k_opt = brentq(objective, -2.0, 2.0, xtol=1e-8)
        return k_opt
    except ValueError:
        return 0.0


def df_to_dict(df):
    """Convert a DataFrame of option data to a simplified dict format."""
    return {
        "forward_price": df["forward_price"].values,
        "tau": df["tau"].values,
        "risk_free_rate": df["risk_free_rate"].values,
        "strike_price": df["strike_price"].values,
        "option_price": df["option_price"].values,
        "log_moneyness": df["log_moneyness"].values,
        "implied_volatility": df["implied_volatility"].values,
        "delta": df["delta"].values,
    }


# ============================================================
# 4. PYTORCH DATASET
# ============================================================
class OptionDataset(Dataset):
    """
    PyTorch Dataset for HyperIV training.

    Each sample corresponds to one date (one IV surface):
      - z: (M, 3) reference set — M contracts with (k, t, sigma)
      - X: (N, 2) query points — all contracts' (k, t)
      - y: (N,)   target values — implied volatilities

    Args:
        df: DataFrame with columns: date, log_moneyness, tau,
            implied_volatility, is_ref (1 for reference contracts)
        N: number of query contracts to sample per date (if sample=True)
        sample: if True, randomly sample N contracts; if False, use all
    """

    def __init__(self, df, N=1024, M=9, sample=True):
        self.N = N
        self.M = M  # Fixed reference set size
        self.sample = sample

        # Group by date
        self.dates = sorted(df["date"].unique())
        self.date_data = {}

        for date in self.dates:
            date_df = df[df["date"] == date]

            # Reference contracts (deduplicated)
            ref_df = date_df[date_df["is_ref"] == 1].drop_duplicates(
                subset=["log_moneyness", "tau"]
            )
            if len(ref_df) == 0:
                continue

            z = ref_df[["log_moneyness", "tau", "implied_volatility"]].values

            # All contracts for training
            all_k = date_df["log_moneyness"].values
            all_t = date_df["tau"].values
            all_sigma = date_df["implied_volatility"].values

            self.date_data[date] = {
                "z": z.astype(np.float32),
                "k": all_k.astype(np.float32),
                "t": all_t.astype(np.float32),
                "sigma": all_sigma.astype(np.float32),
            }

        self.valid_dates = [d for d in self.dates if d in self.date_data]

    def __len__(self):
        return len(self.valid_dates)

    def _pad_or_truncate_ref(self, z):
        """Ensure reference set is exactly M rows by padding or truncating."""
        M_actual = z.shape[0]
        if M_actual == self.M:
            return z
        elif M_actual > self.M:
            # Truncate (keep first M)
            return z[:self.M]
        else:
            # Pad by repeating the last row
            pad = np.tile(z[-1:], (self.M - M_actual, 1))
            return np.concatenate([z, pad], axis=0)

    def __getitem__(self, idx):
        date = self.valid_dates[idx]
        data = self.date_data[date]

        z_raw = data["z"]
        z = self._pad_or_truncate_ref(z_raw)
        z = torch.from_numpy(z)  # (M, 3) — always exactly M rows

        k = data["k"]
        t = data["t"]
        sigma = data["sigma"]

        n_contracts = len(k)

        if self.sample and n_contracts > self.N:
            # Random subsample
            indices = np.random.choice(n_contracts, self.N, replace=False)
        elif n_contracts < self.N:
            # Oversample with replacement to reach exactly N
            indices = np.random.choice(n_contracts, self.N, replace=True)
        else:
            if self.sample:
                indices = np.arange(n_contracts)
            else:
                # For eval: use all but pad/truncate to N for batching
                if n_contracts > self.N:
                    indices = np.arange(self.N)
                else:
                    indices = np.random.choice(n_contracts, self.N, replace=True)

        X = np.stack([k[indices], t[indices]], axis=1)  # (N, 2)
        y = sigma[indices]  # (N,)

        return z, torch.from_numpy(X), torch.from_numpy(y)


# ============================================================
# 5. COLLATE FUNCTION FOR VARIABLE-SIZE SETS
# ============================================================
def hyperiv_collate_fn(batch):
    """
    Custom collate for batches where reference sets may have different sizes.
    Pads reference sets to the max size in the batch.
    """
    z_list, X_list, y_list = zip(*batch)

    # Pad reference sets
    max_M = max(z.shape[0] for z in z_list)
    z_padded = []
    for z in z_list:
        if z.shape[0] < max_M:
            pad = torch.zeros(max_M - z.shape[0], z.shape[1])
            z = torch.cat([z, pad], dim=0)
        z_padded.append(z)

    z_batch = torch.stack(z_padded)
    X_batch = torch.stack(X_list)
    y_batch = torch.stack(y_list)

    return z_batch, X_batch, y_batch
