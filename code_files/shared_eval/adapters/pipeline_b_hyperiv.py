"""Pipeline B — HyperIV (transformer set-encoder -> 337-D omega -> tiny MLP h_omega).

Uses the existing HyperIV implementation from `code_files/pipeline_B_hyperiv/`.

This adapter:
  1. Builds a HyperIV-format dataframe directly from the unified `quotes_dict`
     (BS inversion already done in `unified_dataset` so we just rename columns
     and add `delta` + `is_ref`).
  2. Trains the hyper-net on TRAIN dates only, validates on VAL.
  3. Caches the trained checkpoint, plus per-date weight vectors
     omega_t in R^337 (one inference call per date).
  4. Exposes `eval_at_points` (continuous σ via h_omega) and `eval_grid`
     (12x11 surface for VolGAN).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code_files") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code_files"))

from pipeline_A_isnn.stage4_iv_sampling import bs_normalized_call  # noqa: E402
from pipeline_B_hyperiv.data_util import black_scholes_delta  # noqa: E402
from pipeline_B_hyperiv.hyperiv_util import create_hyperiv_model  # noqa: E402
from pipeline_B_hyperiv.stage1_data_prep import select_reference_set  # noqa: E402
from pipeline_B_hyperiv.stage2_train_hyperiv import train_hyperiv  # noqa: E402
from shared_eval.eval_grid import (  # noqa: E402
    LOG_MONEYNESS_GRID,
    MATURITY_YEARS_GRID,
    GRID_SHAPE,
)


DEFAULTS = dict(
    input_dim=3, hidden_dim=128, num_heads=2, num_layers=2,
    num_epochs=500, batch_size=128, lr=1e-3, N_contracts=1024,
)
QUICK = dict(
    input_dim=3, hidden_dim=64, num_heads=2, num_layers=2,
    num_epochs=5, batch_size=4, lr=1e-3, N_contracts=256,
)


# ---------------------------------------------------------------------------
# Dataframe assembly (unified quotes -> HyperIV format)
# ---------------------------------------------------------------------------

def build_hyperiv_df(quotes_dict: dict, dates) -> pd.DataFrame:
    """Build a long-form DataFrame with columns expected by HyperIV training."""
    rows = []
    for d in dates:
        if d not in quotes_dict:
            continue
        df = quotes_dict[d].copy()
        if len(df) < 9:
            continue
        df["date"] = pd.Timestamp(d).normalize()
        df["tau"] = df["T"]
        df["implied_volatility"] = df["sigma_market"]
        # Compute call delta for the anchor selector.
        df["delta"] = black_scholes_delta(
            K=df["strike"].to_numpy(), t=df["tau"].to_numpy(),
            F=df["fwd_price"].to_numpy(), is_call=1,
            sigma=df["implied_volatility"].to_numpy(),
        )
        # Anchor selection (same routine pipeline_c uses).
        df_for_pick = df.reset_index(drop=True)
        try:
            ref_idx = select_reference_set(df_for_pick)
            ref_idx = list(dict.fromkeys(ref_idx))
        except Exception:
            ref_idx = list(range(min(9, len(df_for_pick))))
        df_for_pick["is_ref"] = 0
        df_for_pick.loc[ref_idx, "is_ref"] = 1
        rows.append(df_for_pick[[
            "date", "log_moneyness", "tau", "implied_volatility",
            "delta", "is_ref",
        ]])
    if not rows:
        return pd.DataFrame(columns=[
            "date", "log_moneyness", "tau", "implied_volatility",
            "delta", "is_ref",
        ])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    quotes_dict: dict,
    train_dates,
    val_dates,
    *,
    quick: bool = False,
    cache_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    hp = QUICK if quick else DEFAULTS

    if verbose:
        print(f"  [pipeline_b] building HyperIV dataframe (train+val)...")
    train_df = build_hyperiv_df(quotes_dict, train_dates)
    val_df = build_hyperiv_df(quotes_dict, val_dates)
    if verbose:
        print(f"  [pipeline_b] train rows={len(train_df)} dates={train_df['date'].nunique()}, "
              f"val rows={len(val_df)} dates={val_df['date'].nunique()}")

    save_dir = None
    if cache_dir is not None:
        save_dir = Path(cache_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_hyperiv(
        train_df, val_df,
        input_dim=hp["input_dim"], hidden_dim=hp["hidden_dim"],
        num_heads=hp["num_heads"], num_layers=hp["num_layers"],
        num_epochs=hp["num_epochs"], batch_size=hp["batch_size"],
        lr=hp["lr"], N_contracts=hp["N_contracts"],
        save_dir=save_dir,
    )
    state = {
        "config": hp,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "history": history,
    }
    if save_dir is not None:
        with open(save_dir / "adapter_state.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    return state


def load(cache_dir: Path | str) -> dict:
    p = Path(cache_dir) / "adapter_state.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_model(state: dict, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = state["config"]
    model, _ = create_hyperiv_model(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        device=device,
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, device


def precompute_omega(state: dict, quotes_dict: dict, dates) -> dict:
    """Compute and cache omega_t = g_theta(Z_t) for each date.

    Returns {date: omega (337,)}.
    """
    df = build_hyperiv_df(quotes_dict, dates)
    model, device = _build_model(state)
    out: dict[pd.Timestamp, np.ndarray] = {}
    for d, g in df.groupby("date"):
        ref = g[g["is_ref"] == 1].drop_duplicates(["log_moneyness", "tau"])
        if len(ref) < 3:
            continue
        z = ref[["log_moneyness", "tau", "implied_volatility"]].to_numpy(np.float32)
        # Pad/truncate to M=9.
        if len(z) > 9:
            z = z[:9]
        elif len(z) < 9:
            z = np.concatenate([z, np.tile(z[-1:], (9 - len(z), 1))], axis=0)
        z_t = torch.from_numpy(z).unsqueeze(0).to(device)
        with torch.no_grad():
            omega = model.get_weights(z_t).squeeze(0).cpu().numpy()
        out[pd.Timestamp(d).normalize()] = omega
    return out


def _eval_h_omega(model, omega: np.ndarray, k: np.ndarray, T: np.ndarray, device) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()
    if len(k) == 0:
        return np.zeros(0)
    omega_t = torch.from_numpy(omega.astype(np.float32)).unsqueeze(0).to(device)
    kt = np.stack([k, T], axis=1).astype(np.float32)
    kt_t = torch.from_numpy(kt).unsqueeze(0).to(device)
    with torch.no_grad():
        sigma = model.forward_from_weights(omega_t, kt_t).squeeze(0).squeeze(-1)
    return sigma.cpu().numpy().astype(np.float64)


def eval_at_points(
    state: dict, omega_dict: dict, date, df_day: pd.DataFrame
) -> np.ndarray:
    d = pd.Timestamp(date).normalize()
    omega = omega_dict.get(d)
    k = df_day["log_moneyness"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    if omega is None:
        return np.full_like(k, np.nan)
    model, device = _build_model(state)
    return _eval_h_omega(model, omega, k, T, device)


def eval_price_at_points(
    state: dict, omega_dict: dict, date, df_day: pd.DataFrame
) -> np.ndarray:
    """Phase-1 price domain: BS-price the predicted IV.

    B lives in IV-space, so we BS-price each (k, T, sigma_pred) using the
    day's per-row F and B from the unified df.
    """
    sigma = eval_at_points(state, omega_dict, date, df_day)
    k = df_day["log_moneyness"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    F = df_day["fwd_price"].to_numpy(dtype=np.float64)
    B = df_day["discount"].to_numpy(dtype=np.float64)
    c_norm = bs_normalized_call(k, T, np.asarray(sigma, dtype=np.float64))
    return c_norm * B * F


def eval_grid(state: dict, omega_dict: dict, date) -> np.ndarray | None:
    d = pd.Timestamp(date).normalize()
    omega = omega_dict.get(d)
    if omega is None:
        return None
    model, device = _build_model(state)
    K, T = np.meshgrid(LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID, indexing="ij")
    iv = _eval_h_omega(model, omega, K.ravel(), T.ravel(), device)
    return np.clip(iv, 0.01, 1.5).reshape(GRID_SHAPE)
