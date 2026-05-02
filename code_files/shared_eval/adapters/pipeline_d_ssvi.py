"""Pipeline D — per-day SSVI fit (no residual hyper-net).

Wraps `code_files/pipeline_D_ssvi/ssvi.py::fit_ssvi` so each test/val/train
day gets its own (a, b, c, η, ρ) parameters fit directly to that day's
arb-filtered market IVs from the unified cache.

State stored per date is just the 5-tuple of params + fit metrics. SSVI's
`ssvi_iv(k, t, param)` is the continuous σ(k, T) used for both Phase-1 (at
market points) and Phase-2 (on the 12x11 grid).
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code_files") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code_files"))

from pipeline_A_isnn.stage4_iv_sampling import bs_normalized_call  # noqa: E402
from pipeline_D_ssvi.ssvi import fit_ssvi, ssvi_iv  # noqa: E402
from shared_eval.eval_grid import (  # noqa: E402
    LOG_MONEYNESS_GRID,
    MATURITY_YEARS_GRID,
    GRID_SHAPE,
)


DEFAULTS = dict(n_restarts=10, atm_weight=3.0)
QUICK = dict(n_restarts=3, atm_weight=3.0)


def train_all(
    quotes_dict: dict,
    dates,
    *,
    quick: bool = False,
    cache_path: Path | str | None = None,
    verbose: bool = True,
    progress_every: int = 50,
) -> dict:
    hp = QUICK if quick else DEFAULTS
    out: dict[pd.Timestamp, dict] = {}
    t0 = time.time()
    for i, d in enumerate(dates):
        if d not in quotes_dict:
            continue
        df = quotes_dict[d]
        if len(df) < 5:
            continue
        try:
            fit = fit_ssvi(
                log_moneyness=df["log_moneyness"].to_numpy(),
                tau=df["T"].to_numpy(),
                iv=df["sigma_market"].to_numpy(),
                n_restarts=hp["n_restarts"],
                atm_weight=hp["atm_weight"],
            )
        except Exception as exc:  # log + skip
            if verbose:
                print(f"  [pipeline_d] {d.date()} fit failed: {exc}")
            continue
        out[pd.Timestamp(d).normalize()] = {
            "param": tuple(map(float, fit.param)),
            "mae": fit.mae,
            "rmse": fit.rmse,
            "n_quotes": fit.n_quotes,
        }
        if verbose and ((i + 1) % progress_every == 0 or i == 0):
            print(f"  [pipeline_d] fit {i+1}/{len(dates)} in {time.time()-t0:.1f}s")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"  [pipeline_d] saved cache: {cache_path}")
    return out


def load(cache_path: Path | str) -> dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def eval_at_points(state: dict, date, df_day: pd.DataFrame) -> np.ndarray:
    d = pd.Timestamp(date).normalize()
    entry = state.get(d)
    k = df_day["log_moneyness"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    if entry is None:
        return np.full_like(k, np.nan)
    return ssvi_iv(k, T, entry["param"])


def eval_price_at_points(state: dict, date, df_day: pd.DataFrame) -> np.ndarray:
    """Phase-1 price domain: BS-price SSVI's IV using per-row F, B from df."""
    sigma = eval_at_points(state, date, df_day)
    k = df_day["log_moneyness"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    F = df_day["fwd_price"].to_numpy(dtype=np.float64)
    B = df_day["discount"].to_numpy(dtype=np.float64)
    c_norm = bs_normalized_call(k, T, np.asarray(sigma, dtype=np.float64))
    return c_norm * B * F


def eval_grid(state: dict, date) -> np.ndarray | None:
    d = pd.Timestamp(date).normalize()
    entry = state.get(d)
    if entry is None:
        return None
    K, T = np.meshgrid(LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID, indexing="ij")
    iv = ssvi_iv(K, T, entry["param"])
    return np.clip(iv, 0.01, 1.5).reshape(GRID_SHAPE)
