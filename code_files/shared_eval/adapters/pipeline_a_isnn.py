"""Pipeline A — per-day ISNN-2 fit on call prices, then BS-invert.

Wraps `code_files/pipeline_A_isnn/stage3_isnn.py` to expose:
  - `train_all(quotes_dict, dates, ...)` — fit one ISNN per day, cache to disk.
  - `load(cache_path)` — restore a previously trained run.
  - `eval_at_points(state, k, T)` — Phase-1 entry: evaluate the price net
    at (k, T) and BS-invert to IV at those exact market quote points.
  - `eval_grid(state)` — Phase-2 entry: produce the (12, 11) IV grid.

A "state" here means a dict of trained per-day models in memory plus the
ISNNAdapter wrapper. The pickle cache stores only state-dicts (CPU) +
processor norm factors so we can reload without re-running training.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code_files") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code_files"))

from pipeline_A_isnn.stage3_isnn import (  # noqa: E402
    ISNN2_OptionSurface,
    fit_surface_for_date,
)
from pipeline_A_isnn.stage4_iv_sampling import (  # noqa: E402
    bs_normalized_call,
    implied_vol_brentq_fallback,
    implied_vol_newton,
)
from shared_eval.eval_grid import (  # noqa: E402
    LOG_MONEYNESS_GRID,
    MATURITY_YEARS_GRID,
    GRID_SHAPE,
)


# ---------------------------------------------------------------------------
# Default training hyperparameters. `--quick` mode shrinks epochs.
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    epochs=2000,
    lr=0.005,
    hidden_dim=128,
    num_layers=3,
)
QUICK = dict(
    epochs=200,
    lr=0.005,
    hidden_dim=64,
    num_layers=3,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _prep_df_for_isnn(df: pd.DataFrame) -> pd.DataFrame:
    """Stage-3 expects a `moussa_price` column; the unified cache stores it as `mid`."""
    out = df.copy()
    out["moussa_price"] = out["mid"]
    return out


def train_all(
    quotes_dict: dict,
    dates,
    *,
    quick: bool = False,
    cache_path: Path | str | None = None,
    verbose: bool = True,
    progress_every: int = 25,
) -> dict:
    """Fit one ISNN per date.

    Args:
        quotes_dict: unified `quotes_dict` from `unified_dataset`.
        dates: iterable of pd.Timestamps to train on.
        quick: shrink epochs for verification.
        cache_path: if given, pickle a CPU state-dict cache here.

    Returns a dict {date -> {"state_dict", "hidden_dim", "num_layers"}}.
    """
    hp = QUICK if quick else DEFAULTS

    out = {}
    t0 = time.time()
    for i, d in enumerate(dates):
        if d not in quotes_dict:
            continue
        df = _prep_df_for_isnn(quotes_dict[d])
        if len(df) < 5:
            continue
        model, _processor, _losses = fit_surface_for_date(
            date_key=str(pd.Timestamp(d).date()),
            dataframe=df,
            epochs=hp["epochs"],
            lr=hp["lr"],
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            verbose=False,
        )
        out[pd.Timestamp(d).normalize()] = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "hidden_dim": hp["hidden_dim"],
            "num_layers": hp["num_layers"],
        }
        if verbose and ((i + 1) % progress_every == 0 or i == 0):
            print(f"  [pipeline_a] fit {i+1}/{len(dates)} in {time.time()-t0:.1f}s")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"  [pipeline_a] saved cache: {cache_path}")
    return out


def load(cache_path: Path | str) -> dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_model(entry: dict) -> ISNN2_OptionSurface:
    m = ISNN2_OptionSurface(
        hidden_dim=entry["hidden_dim"], num_layers=entry["num_layers"]
    )
    m.load_state_dict(entry["state_dict"])
    m.eval()
    return m


def _isnn_to_iv(model: ISNN2_OptionSurface, k: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Query the ISNN at (k=log(K/F), T) and invert BS to get sigma.

    The ISNN takes (K/F, T_years) -> C_norm = C/(B*F).
    """
    k = np.asarray(k, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    kf = np.exp(k)

    x0 = torch.tensor(kf, dtype=torch.float32).unsqueeze(1)
    t0 = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        c_norm = model(x0, t0).squeeze(1).numpy().astype(np.float64)

    intrinsic = np.maximum(0.0, 1.0 - kf)
    c_clamped = np.clip(c_norm, intrinsic + 1e-8, 1.0 - 1e-8)
    has_tv = (c_clamped - intrinsic) > 1e-7

    iv = np.full_like(c_clamped, np.nan)
    if has_tv.any():
        iv[has_tv] = implied_vol_newton(c_clamped[has_tv], k[has_tv], T[has_tv])

    nan_mask = np.isnan(iv) & has_tv
    if nan_mask.any():
        for idx in np.where(nan_mask)[0]:
            iv[idx] = implied_vol_brentq_fallback(c_clamped[idx], k[idx], T[idx])
    return iv


def eval_at_points(state: dict, date, df_day: pd.DataFrame) -> np.ndarray:
    """Phase-1: evaluate σ_pipeline_A at the market quote points of `df_day`.

    df_day must have columns `log_moneyness` and `T` (per the unified format).
    """
    d = pd.Timestamp(date).normalize()
    entry = state.get(d)
    k = df_day["log_moneyness"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    if entry is None:
        return np.full_like(k, np.nan)
    model = _build_model(entry)
    return _isnn_to_iv(model, k, T)


def eval_price_at_points(state: dict, date, df_day: pd.DataFrame) -> np.ndarray:
    """Phase-1 price domain: native call price from the ISNN.

    The ISNN takes (K/F, T) and emits C_norm = C / (B*F). We multiply back
    by B*F (per-row from df_day) to recover the raw call price comparable
    to df_day['mid'].
    """
    d = pd.Timestamp(date).normalize()
    entry = state.get(d)
    k = df_day["log_moneyness"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    F = df_day["fwd_price"].to_numpy(dtype=np.float64)
    B = df_day["discount"].to_numpy(dtype=np.float64)
    if entry is None:
        return np.full_like(k, np.nan)
    model = _build_model(entry)
    kf = np.exp(k)
    x0 = torch.tensor(kf, dtype=torch.float32).unsqueeze(1)
    t0 = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        c_norm = model(x0, t0).squeeze(1).numpy().astype(np.float64)
    return c_norm * B * F


def eval_grid(state: dict, date) -> np.ndarray | None:
    """Phase-2: return the standard 12x11 IV grid for `date`, or None if missing."""
    d = pd.Timestamp(date).normalize()
    entry = state.get(d)
    if entry is None:
        return None
    model = _build_model(entry)
    K, T = np.meshgrid(LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID, indexing="ij")
    iv_flat = _isnn_to_iv(model, K.ravel(), T.ravel())

    grid = iv_flat.reshape(GRID_SHAPE)
    if np.isnan(grid).any():
        # Simple iterative neighbour fill (matches stage4 behaviour).
        for _ in range(max(GRID_SHAPE)):
            nan = np.isnan(grid)
            if not nan.any():
                break
            filled = grid.copy()
            for i, j in zip(*np.where(nan)):
                neigh = []
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < grid.shape[0] and 0 <= jj < grid.shape[1] \
                            and not np.isnan(grid[ii, jj]):
                        neigh.append(grid[ii, jj])
                if neigh:
                    filled[i, j] = float(np.mean(neigh))
            grid = filled
        if np.isnan(grid).any():
            grid = np.where(np.isnan(grid), np.nanmean(grid), grid)
    return np.clip(grid, 0.01, 1.5)
