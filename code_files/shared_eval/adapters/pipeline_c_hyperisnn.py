"""Pipeline C — HyperISNN v3 (hypernet -> ISNN target with hard arb-free constraints).

Wraps `code_files/pipeline_C_hyperisnn/`. The HyperISNN model:
  - takes a per-day reference set Z (random subset of that day's quotes) plus
    a context vector ctx (centred forward curve + rate),
  - emits ISNN-2 weights via a transformer hypernet,
  - the ISNN takes (m=K/S, T) and produces C/S which we BS-invert to IV.

We rebuild "prepared days" from the unified `quotes_dict` + `market_data` so we
do NOT depend on the legacy `data/shared/filtered_dict.pkl` cache.
"""

from __future__ import annotations

import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code_files") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code_files"))

from pipeline_C_hyperisnn.bs_utils import bs_vega, implied_vol  # noqa: E402
from pipeline_C_hyperisnn.config import Config  # noqa: E402
from pipeline_C_hyperisnn.model import HyperISNN  # noqa: E402
from pipeline_C_hyperisnn.trainer import (  # noqa: E402
    Trainer, collate, sample_episode, to_device,
)
from pipeline_C_hyperisnn.losses import build_loss  # noqa: E402
from shared_eval.eval_grid import (  # noqa: E402
    LOG_MONEYNESS_GRID,
    MATURITY_YEARS_GRID,
    GRID_SHAPE,
)


DEFAULT_OVERRIDES = dict(n_steps=10000, warmup_steps=500, batch_size=32)
QUICK_OVERRIDES = dict(n_steps=200, warmup_steps=50, batch_size=8)


# ---------------------------------------------------------------------------
# Build "prepared days" directly from unified payload
# ---------------------------------------------------------------------------

def _build_prepared(payload: dict, dates) -> list[dict]:
    """Construct the v3 prepared-day dict list from the unified payload.

    Mirrors `pipeline_C_hyperisnn.data.prepare_day` but pulls from the
    unified per-quote dataset that already passed the Moussa filter +
    BS-inversion in `unified_dataset.py`.
    """
    quotes = payload["quotes_dict"]
    market = payload["market_data"]
    St_df = market["St_df"]
    fwd_df = market["fwd_df"]
    discount_df = market["discount_df"]

    # Match the discount-curve column the v3 trainer uses (5th = ~90d).
    out: list[dict] = []
    for d in dates:
        d = pd.Timestamp(d).normalize()
        if d not in quotes:
            continue
        df = quotes[d]
        if len(df) < 30:
            continue

        # Spot from St_df (handle missing date by nearest).
        st_idx = St_df.index.get_indexer([d], method="nearest")[0]
        if st_idx < 0:
            continue
        S_row = St_df.iloc[st_idx]
        S = float(S_row.iloc[0]) if "Spot" not in S_row.index else float(S_row["Spot"])

        fwd_idx = fwd_df.index.get_indexer([d], method="nearest")[0]
        disc_idx = discount_df.index.get_indexer([d], method="nearest")[0]
        if fwd_idx < 0 or disc_idx < 0:
            continue
        fwd_curve = fwd_df.iloc[fwd_idx].to_numpy(dtype=np.float32)
        disc_curve = discount_df.iloc[disc_idx].to_numpy(dtype=np.float32)
        if disc_curve.shape[0] >= 6 and disc_curve[5] > 0:
            r = -float(np.log(disc_curve[5])) / (90.0 / 365.0)
        else:
            r = 0.02

        K = df["strike"].to_numpy(dtype=np.float64)
        T = df["T"].to_numpy(dtype=np.float64)
        price = df["mid"].to_numpy(dtype=np.float64)
        iv = df["sigma_market"].to_numpy(dtype=np.float64)
        vega = bs_vega(S, K, T, r, iv).astype(np.float32)

        fwd = df["fwd_price"].to_numpy(dtype=np.float32)
        disc = df["discount"].to_numpy(dtype=np.float32)

        out.append(dict(
            date=d,
            S=float(S),
            r=float(r),
            fwd_curve=fwd_curve,
            K=K.astype(np.float32),
            T=T.astype(np.float32),
            price=price.astype(np.float32),
            m=(K / fwd).astype(np.float32),
            logm=np.log(K / fwd).astype(np.float32),
            C_norm=(price / (disc * fwd)).astype(np.float32),
            iv=iv.astype(np.float32),
            vega=vega,
        ))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    payload: dict,
    train_dates,
    val_dates,
    *,
    quick: bool = False,
    cache_dir: Path | str | None = None,
    loss_name: str = "hybrid",
    verbose: bool = True,
) -> dict:
    cfg = Config()
    overrides = QUICK_OVERRIDES if quick else DEFAULT_OVERRIDES
    cfg.train.n_steps = overrides["n_steps"]
    cfg.train.warmup_steps = overrides["warmup_steps"]
    cfg.train.batch_size = overrides["batch_size"]
    if cache_dir is not None:
        cfg.train.checkpoint_dir = Path(cache_dir)
        cfg.train.output_file = str(Path(cache_dir) / "train.log")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  [pipeline_c] preparing days...")
    train_data = _build_prepared(payload, train_dates)
    val_data = _build_prepared(payload, val_dates)
    if verbose:
        print(f"  [pipeline_c] train={len(train_data)} val={len(val_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    model = HyperISNN(cfg)
    loss_fn = build_loss(loss_name, cfg.loss)
    trainer = Trainer(
        cfg, model, loss_fn, train_data, val_data, device,
        log_fn=(print if verbose else (lambda *_a, **_k: None)),
    )
    trainer.train()

    state = {
        "config": cfg,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "history": trainer.history,
    }
    if cache_dir is not None:
        with open(Path(cache_dir) / "adapter_state.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    return state


def load(cache_dir: Path | str) -> dict:
    with open(Path(cache_dir) / "adapter_state.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_model(state: dict, device=None) -> tuple[HyperISNN, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg: Config = deepcopy(state["config"])
    model = HyperISNN(cfg).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, device


def _episode_inputs(day: dict, cfg_hyper, device, *, seed: int = 0):
    """Build the (q, mask, ctx) tensors for a single test day from ALL its quotes.

    Uses up to n_ctx_max contracts as context. We don't sample stochastically
    at eval time — we just take the first n_ctx_max (or fewer) anchors after a
    deterministic shuffle.
    """
    rng = np.random.default_rng(seed)
    n_q = len(day["K"])
    n_ctx = min(cfg_hyper.n_ctx_max, n_q)
    perm = rng.permutation(n_q)
    c_idx = perm[:n_ctx]

    q_pad = np.zeros((1, cfg_hyper.n_ctx_max, 3), dtype=np.float32)
    cmask = np.zeros((1, cfg_hyper.n_ctx_max), dtype=bool)
    ctx = np.zeros((1, 9), dtype=np.float32)

    q_pad[0, :n_ctx, 0] = day["logm"][c_idx]
    q_pad[0, :n_ctx, 1] = day["T"][c_idx]
    q_pad[0, :n_ctx, 2] = day["C_norm"][c_idx]
    cmask[0, :n_ctx] = True
    fwd8 = day["fwd_curve"][:8]
    if fwd8.shape[0] < 8:
        fwd8 = np.concatenate([fwd8, np.full(8 - fwd8.shape[0], fwd8[-1])])
    ctx[0, :8] = (fwd8 / day["S"] - 1.0) * 10.0
    ctx[0, 8] = day["r"] * 10.0

    return (
        torch.from_numpy(q_pad).to(device),
        torch.from_numpy(cmask).to(device),
        torch.from_numpy(ctx).to(device),
    )


def _predict_cnorm(
    model: HyperISNN, day: dict, m_query: np.ndarray, T_query: np.ndarray, device
) -> np.ndarray:
    """Run the hypernet+ISNN and return raw C/S (network's native output)."""
    q, mask, ctx = _episode_inputs(day, model.cfg.hyper, device)
    m_t = torch.from_numpy(m_query.astype(np.float32)).view(-1, 1).to(device)
    T_t = torch.from_numpy(T_query.astype(np.float32)).view(-1, 1).to(device)
    r_t = torch.full_like(T_t, float(day["r"]))
    with torch.no_grad():
        c_norm = model.predict_surface(q, mask, ctx, m_t, T_t, r_t).squeeze(-1).cpu().numpy()
    return c_norm


def _predict_iv(
    model: HyperISNN, day: dict, m_query: np.ndarray, T_query: np.ndarray, device, fwd: np.ndarray, disc: np.ndarray
) -> np.ndarray:
    """Run the hypernet+ISNN, then BS-invert the predicted C/S to IV."""
    c_norm = _predict_cnorm(model, day, m_query, T_query, device)
    price = c_norm * disc * fwd
    K = fwd * m_query
    iv = implied_vol(price.astype(np.float64), float(day["S"]), K, T_query, float(day["r"]))
    return iv


def precompute_days(state: dict, payload: dict, dates) -> dict:
    """Build a dict {date -> prepared-day} for inference reuse."""
    return {pd.Timestamp(d["date"]).normalize(): d
            for d in _build_prepared(payload, dates)}


def eval_at_points(
    state: dict, days: dict, date, df_day: pd.DataFrame
) -> np.ndarray:
    d = pd.Timestamp(date).normalize()
    day = days.get(d)
    nq = len(df_day)
    if day is None:
        return np.full(nq, np.nan)
    model, device = _build_model(state)
    K = df_day["strike"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    fwd = df_day["fwd_price"].to_numpy(dtype=np.float64)
    disc = df_day["discount"].to_numpy(dtype=np.float64)
    m = K / fwd
    return _predict_iv(model, day, m, T, device, fwd, disc)


def eval_price_at_points(
    state: dict, days: dict, date, df_day: pd.DataFrame
) -> np.ndarray:
    """Phase-1 price domain: native call price from the HyperISNN.

    HyperISNN emits C/S; we multiply by S to recover the raw price
    comparable to df_day['mid'].
    """
    d = pd.Timestamp(date).normalize()
    day = days.get(d)
    nq = len(df_day)
    if day is None:
        return np.full(nq, np.nan)
    model, device = _build_model(state)
    K = df_day["strike"].to_numpy(dtype=np.float64)
    T = df_day["T"].to_numpy(dtype=np.float64)
    fwd = df_day["fwd_price"].to_numpy(dtype=np.float64)
    disc = df_day["discount"].to_numpy(dtype=np.float64)
    m = K / fwd
    c_norm = _predict_cnorm(model, day, m, T, device)
    return c_norm * disc * fwd


def eval_grid(state: dict, days: dict, date) -> np.ndarray | None:
    d = pd.Timestamp(date).normalize()
    day = days.get(d)
    if day is None:
        return None
    model, device = _build_model(state)
    # Grid is in (k=log(K/F), T). v3's ISNN expects (m=K/S, T). Convert via
    # the day's mean fwd_price (we don't have a per-T fwd here for the grid).
    # Use the day's at-the-money fwd_price ~ S * exp(0)*(F/S). Simpler: use the
    # 90d forward as the canonical F for converting k -> m. This is acceptable
    # because the 12x11 grid's IVs are then bilinearly interpolated downstream
    # anyway, and the absolute K/S placement only enters via the ISNN's
    # nonlinear smile — small displacement vs daily-fwd-curve effect.
    F = float(day["fwd_curve"][5]) if day["fwd_curve"].shape[0] >= 6 else float(day["fwd_curve"][-1])
    Kg, Tg = np.meshgrid(LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID, indexing="ij")
    K_abs = F * np.exp(Kg.ravel())
    # Interpolate Fwd curve and compute discount for the grid. 
    # For a quick evaluation, just use the canonical F and canonical discount.
    r_val = float(day["r"])
    disc = np.exp(-r_val * Tg.ravel())
    fwd_vec = np.full_like(Tg.ravel(), F)
    m = K_abs / fwd_vec
    iv = _predict_iv(model, day, m, Tg.ravel(), device, fwd_vec, disc)
    iv = np.where(np.isfinite(iv), iv, np.nan)
    grid = iv.reshape(GRID_SHAPE)
    if np.isnan(grid).any():
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
