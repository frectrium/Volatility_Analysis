"""Data loading and per-day preprocessing.

Compatible with the v2 cache (`hyperisnn_prepared_cache.pkl`) so that
re-running v3 does not require re-preprocessing the full dataset.

A "prepared day" is a dict with keys:
    date, S, r, fwd_curve, K, T, price, m, logm, C_norm, iv, vega
"""

import pickle
import time
from pathlib import Path
from typing import List

import numpy as np

from .bs_utils import bs_vega, implied_vol
from .config import DataConfig


def prepare_day(date, df, market, cfg: DataConfig):
    """Build one prepared-day dict, or None if the day is unusable."""
    if cfg.price_col not in df.columns:
        return None
    S = float(df["Spot"].iloc[0])
    K = df["strike"].values.astype(np.float64)
    T = df["maturity_days"].values.astype(np.float64) / 365.0
    price = df[cfg.price_col].values.astype(np.float64)

    mask = (
        (T >= cfg.maturity_lo_days / 365.0)
        & (T <= cfg.maturity_hi_days / 365.0)
        & (K / S >= cfg.moneyness_lo)
        & (K / S <= cfg.moneyness_hi)
        & (price > 0.01)
        & (price < S)
        & np.isfinite(price)
    )
    if mask.sum() < cfg.min_quotes_per_day:
        return None
    K, T, price = K[mask], T[mask], price[mask]

    fwd_curve = market["fwd_df"].loc[date].values.astype(np.float32)
    disc_curve = market["discount_df"].loc[date].values.astype(np.float32)
    # 90-day discount factor → annualized risk-free rate
    if disc_curve[5] > 0:
        r = -float(np.log(disc_curve[5])) / (90.0 / 365.0)
    else:
        r = 0.02

    iv = implied_vol(price, S, K, T, r)
    iv_clean = np.where(np.isnan(iv), 0.20, iv)
    vega = bs_vega(S, K, T, r, iv_clean).astype(np.float32)

    return dict(
        date=date,
        S=float(S),
        r=float(r),
        fwd_curve=fwd_curve,
        K=K.astype(np.float32),
        T=T.astype(np.float32),
        price=price.astype(np.float32),
        m=(K / S).astype(np.float32),
        logm=np.log(K / S).astype(np.float32),
        C_norm=(price / S).astype(np.float32),
        iv=iv.astype(np.float32),
        vega=vega,
    )


def load_or_build_cache(cfg: DataConfig, log_fn=print) -> List[dict]:
    """Load the cache if it exists, else build from raw pickles and cache it."""
    if cfg.cache_path.exists():
        log_fn(f"Loading cached preprocessed data from {cfg.cache_path}")
        with open(cfg.cache_path, "rb") as f:
            return pickle.load(f)

    log_fn(f"Loading {cfg.data_path} ...")
    with open(cfg.data_path, "rb") as f:
        data_dict = pickle.load(f)
    with open(cfg.market_path, "rb") as f:
        market = pickle.load(f)
    log_fn(f"  {len(data_dict)} dates")

    dates = sorted(data_dict.keys())
    log_fn("Preprocessing ...")
    t0 = time.time()
    prepared = []
    for i, d in enumerate(dates):
        out = prepare_day(d, data_dict[d], market, cfg)
        if out is not None:
            prepared.append(out)
        if (i + 1) % 250 == 0:
            log_fn(f"  ... {i+1}/{len(dates)}")
    log_fn(f"  done in {time.time()-t0:.1f}s — kept {len(prepared)}/{len(dates)} days")
    with open(cfg.cache_path, "wb") as f:
        pickle.dump(prepared, f)
    return prepared


def split_chronologically(prepared: List[dict], cfg: DataConfig):
    """Chronological train / val / test split (no leakage)."""
    n = len(prepared)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    return (
        prepared[:n_train],
        prepared[n_train:n_train + n_val],
        prepared[n_train + n_val:],
    )
