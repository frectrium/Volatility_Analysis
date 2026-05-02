"""VolGAN forecaster adapter.

Wraps the upstream `volgan/VolGAN.py` (cloned from
github.com/milenavuletic/VolGAN) so we can train one VolGAN per pipeline
on its own TRAIN-set 12x11 IV surfaces and predict next-day IV grids.

We reuse only the `Generator` and `Discriminator` modules from upstream;
data preparation and the train loop are reimplemented here so we can drive
the model from our cached `surfaces[date] = (12, 11) ndarray` dict + the
shared side-info (logret lags, RV21).

Tensor conventions (Ngrid = 12 * 11 = 132):
- Surfaces are flattened with maturity outer / moneyness inner:
    flat = grid.T.ravel()
    grid = flat.reshape(11, 12).T
  This matches upstream `m_seq`/`matrix_m` semantics where consecutive
  blocks of length `lk` share a maturity.

Condition vector (per day t):
    [logret_{t-1}, logret_{t-2}, rv21_{t-1}, log_iv_{t-1, flat (Ngrid)}]
True vector:
    [logret_{t}_annualised, log_iv_{t, flat} - log_iv_{t-1, flat}]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "volgan") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "volgan"))

# Upstream VolGAN.py imports a few packages we don't need (pandas_datareader,
# yfinance, statsmodels). Stub them so the module loads on minimal envs.
import types as _types  # noqa: E402
for _stub_name in (
    "pandas_datareader", "pandas_datareader.data",
    "yfinance",
    "statsmodels", "statsmodels.tsa", "statsmodels.tsa.stattools",
):
    if _stub_name not in sys.modules:
        _m = _types.ModuleType(_stub_name)
        if _stub_name == "statsmodels.tsa.stattools":
            _m.acf = lambda *a, **k: None
            _m.pacf = lambda *a, **k: None
        if _stub_name == "pandas_datareader":
            _m.data = _types.ModuleType("pandas_datareader.data")
        sys.modules[_stub_name] = _m

# Upstream uses `from scipy import arange, array, exp` (deprecated, removed
# in SciPy >=1.0). Inject the numpy aliases so the import succeeds without
# modifying upstream.
import scipy as _scipy  # noqa: E402
import numpy as _np_for_scipy  # noqa: E402
for _name in ("arange", "array", "exp"):
    if not hasattr(_scipy, _name):
        setattr(_scipy, _name, getattr(_np_for_scipy, _name))

# Upstream Generator/Discriminator + arb-penalty matrices.
import VolGAN as _vg  # noqa: E402

from shared_eval.eval_grid import (  # noqa: E402
    LOG_MONEYNESS_GRID,
    MATURITY_YEARS_GRID,
    N_MONEYNESS,
    N_MATURITY,
    GRID_SHAPE,
)


LK = N_MONEYNESS  # 12
LT = N_MATURITY   # 11
NGRID = LK * LT   # 132

DEFAULTS = dict(
    noise_dim=16, hidden_dim=8, n_epochs=2000,
    lr_g=1e-4, lr_d=1e-4, batch_size=100,
    alpha=0.01, beta=0.01,                # fixed smoothness penalties
)
QUICK = dict(
    noise_dim=16, hidden_dim=8, n_epochs=50,
    lr_g=1e-4, lr_d=1e-4, batch_size=8,
    alpha=0.01, beta=0.01,
)


# ---------------------------------------------------------------------------
# Surface flatten / unflatten
# ---------------------------------------------------------------------------

def grid_flatten(grid: np.ndarray) -> np.ndarray:
    if grid.shape != GRID_SHAPE:
        raise ValueError(f"expected {GRID_SHAPE}, got {grid.shape}")
    return grid.T.ravel().astype(np.float32)


def grid_unflatten(flat: np.ndarray) -> np.ndarray:
    return np.asarray(flat).reshape(LT, LK).T


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_sequences(
    surfaces: dict,
    side_info: pd.DataFrame,
    dates,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Build (true, condition, used_dates) for the consecutive-day pairs in `dates`.

    Skips days where the previous day is missing or has any NaN-side-info.
    """
    dates = sorted(pd.Timestamp(d).normalize() for d in dates)
    side = side_info.reindex(pd.DatetimeIndex(dates))

    rows_true, rows_cond, kept = [], [], []
    for i in range(1, len(dates)):
        d = dates[i]; dm1 = dates[i - 1]
        if d not in surfaces or dm1 not in surfaces:
            continue
        if d not in side.index or dm1 not in side.index:
            continue
        row_t = side.loc[d]
        # condition uses lag1, lag2, rv21_lag1 evaluated at d (so refers to t-1 etc.)
        if not np.isfinite([row_t["logret_lag1"], row_t["logret_lag2"], row_t["rv21_lag1"]]).all():
            continue
        log_iv_tm1 = np.log(np.clip(grid_flatten(surfaces[dm1]), eps, None))
        log_iv_t = np.log(np.clip(grid_flatten(surfaces[d]), eps, None))
        log_iv_inc = log_iv_t - log_iv_tm1

        cond = np.concatenate([
            [float(row_t["logret_lag1"])],
            [float(row_t["logret_lag2"])],
            [float(row_t["rv21_lag1"])],
            log_iv_tm1,
        ]).astype(np.float32)
        # true: annualised logret_t, increment in log iv
        ret_t = float(row_t["logret_lag1"])  # placeholder; we rebuild below
        # Actually logret_t is not directly in side_info row at d (which holds
        # logret_lag1=logret_{d-1}). We need logret_d. Approximation: we don't
        # use logret_t in our forecasting metrics, so we set 0 here. The
        # generator's first output column is the spot-return prediction; we
        # ignore it downstream.
        true = np.concatenate([[0.0], log_iv_inc]).astype(np.float32)

        rows_true.append(true)
        rows_cond.append(cond)
        kept.append(d)

    if not rows_true:
        return (np.zeros((0, 1 + NGRID), dtype=np.float32),
                np.zeros((0, 3 + NGRID), dtype=np.float32),
                [])
    return (np.stack(rows_true), np.stack(rows_cond), kept)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_models(noise_dim: int, cond_dim: int, true_dim: int, hidden_dim: int, device):
    gen = _vg.Generator(
        noise_dim=noise_dim, cond_dim=cond_dim, hidden_dim=hidden_dim,
        output_dim=true_dim,
    ).to(device)
    disc = _vg.Discriminator(
        in_dim=cond_dim + true_dim, hidden_dim=hidden_dim,
    ).to(device)
    return gen, disc


def _smoothness_matrices(device):
    """Build the m-direction and t-direction first-difference matrices used
    for smoothness penalties on the flattened surface (length NGRID)."""
    moneyness_t = torch.tensor(LOG_MONEYNESS_GRID, dtype=torch.float, device=device)
    tau_t = torch.tensor(MATURITY_YEARS_GRID, dtype=torch.float, device=device)

    matrix_t = torch.zeros((NGRID, NGRID), dtype=torch.float, device=device)
    for i in range(NGRID - 1):
        matrix_t[i, i] = -1.0
        matrix_t[i, i + 1] = 1.0
    t_seq = torch.zeros((tau_t.shape[0],), dtype=torch.float, device=device)
    for i in range(tau_t.shape[0] - 1):
        t_seq[i] = 1.0 / max((tau_t[i + 1] - tau_t[i]).item() ** 2, 1e-12)
    tsq = t_seq.repeat(LK).unsqueeze(0)

    matrix_m = torch.zeros((NGRID - LK, NGRID), dtype=torch.float, device=device)
    for i in range(NGRID - LK):
        matrix_m[i, i] = -1.0
        matrix_m[i, i + LK] = 1.0
    m_seq = torch.zeros((LK * (LT - 1),), dtype=torch.float, device=device)
    for i in range(moneyness_t.shape[0] - 1):
        m_seq[i * LK:(i + 1) * LK] = 1.0 / max(
            (moneyness_t[i + 1] - moneyness_t[i]).item() ** 2, 1e-12
        )
    return matrix_m, matrix_t, m_seq, tsq


def train(
    surfaces: dict,
    side_info: pd.DataFrame,
    train_dates,
    val_dates,
    *,
    quick: bool = False,
    cache_path: Path | str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    hp = QUICK if quick else DEFAULTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    true_tr, cond_tr, _ = build_sequences(surfaces, side_info, train_dates)
    if true_tr.shape[0] == 0:
        raise RuntimeError("VolGAN train: no consecutive-day pairs assembled.")
    if verbose:
        print(f"  [volgan] train pairs={true_tr.shape[0]}, cond_dim={cond_tr.shape[1]}, true_dim={true_tr.shape[1]}")

    cond = torch.from_numpy(cond_tr).to(device)
    true = torch.from_numpy(true_tr).to(device)
    n_train = true.shape[0]

    gen, disc = _build_models(
        noise_dim=hp["noise_dim"], cond_dim=cond.shape[1],
        true_dim=true.shape[1], hidden_dim=hp["hidden_dim"], device=device,
    )
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=hp["lr_g"])
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=hp["lr_d"])
    bce = nn.BCELoss()

    matrix_m, matrix_t, m_seq, tsq = _smoothness_matrices(device)
    alpha, beta = hp["alpha"], hp["beta"]

    n_batches = max(1, n_train // hp["batch_size"])
    for epoch in range(hp["n_epochs"]):
        perm = torch.randperm(n_train, device=device)
        cond_p = cond[perm]; true_p = true[perm]
        epoch_g = 0.0; epoch_d = 0.0
        for b in range(n_batches):
            lo = b * hp["batch_size"]
            hi = min(lo + hp["batch_size"], n_train)
            c_b = cond_p[lo:hi]
            t_b = true_p[lo:hi]
            cur = c_b.shape[0]

            # --- discriminator
            disc_opt.zero_grad()
            noise = torch.randn(cur, hp["noise_dim"], device=device)
            fake = gen(noise, c_b).detach()
            real_pair = torch.cat([c_b, t_b], dim=-1)
            fake_pair = torch.cat([c_b, fake], dim=-1)
            d_real = disc(real_pair)
            d_fake = disc(fake_pair)
            d_loss = 0.5 * (bce(d_real, torch.ones_like(d_real))
                            + bce(d_fake, torch.zeros_like(d_fake)))
            d_loss.backward()
            disc_opt.step()

            # --- generator
            gen_opt.zero_grad()
            noise = torch.randn(cur, hp["noise_dim"], device=device)
            fake = gen(noise, c_b)
            fake_pair = torch.cat([c_b, fake], dim=-1)
            d_fake = disc(fake_pair)
            surface_past = c_b[:, 3:]                       # log_iv_{t-1}
            fake_surface = fake[:, 1:] + surface_past        # log_iv_t (predicted)

            # Smoothness penalties on log-IV surface (matches upstream).
            pen_m = ((matrix_m @ fake_surface.t()) ** 2).t()
            pen_t = ((matrix_t @ fake_surface.t()) ** 2).t()
            pen_m_total = (m_seq.unsqueeze(0) * pen_m).sum(dim=1).mean()
            pen_t_total = (tsq * pen_t).sum(dim=1).mean()

            g_loss = (bce(d_fake, torch.ones_like(d_fake))
                      + alpha * pen_m_total + beta * pen_t_total)
            g_loss.backward()
            gen_opt.step()

            epoch_g += float(g_loss.item())
            epoch_d += float(d_loss.item())

        if verbose and ((epoch + 1) % max(1, hp["n_epochs"] // 10) == 0 or epoch == 0):
            print(f"  [volgan] epoch {epoch+1}/{hp['n_epochs']} "
                  f"D={epoch_d/n_batches:.4f} G={epoch_g/n_batches:.4f}")

    state = {
        "config": hp,
        "gen": {k: v.detach().cpu() for k, v in gen.state_dict().items()},
        "disc": {k: v.detach().cpu() for k, v in disc.state_dict().items()},
        "cond_dim": cond.shape[1],
        "true_dim": true.shape[1],
    }
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"  [volgan] saved {cache_path}")
    return state


def load(cache_path: Path | str) -> dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_gen(state: dict, device):
    cfg = state["config"]
    gen = _vg.Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=state["cond_dim"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=state["true_dim"],
    ).to(device)
    gen.load_state_dict(state["gen"])
    gen.eval()
    return gen


def predict_grid(
    state: dict,
    surfaces: dict,
    side_info: pd.DataFrame,
    date,
    *,
    n_samples: int = 1,
    eps: float = 1e-6,
    seed: int | None = None,
) -> np.ndarray | None:
    """Predict the (12, 11) IV grid for `date` given pipeline surfaces[date-1].

    Stochastic by default — one noise draw per call (`n_samples=1`,
    `seed=None` so each call returns a different sample). Set
    `n_samples > 1` to draw multiple realisations and average; set
    `seed` to make a specific call reproducible. Returns None if the
    previous day's surface or side-info is missing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = pd.Timestamp(date).normalize()
    side = side_info.reindex(pd.DatetimeIndex([d]))
    if d not in side.index:
        return None
    row_t = side.loc[d]
    if not np.isfinite([row_t["logret_lag1"], row_t["logret_lag2"], row_t["rv21_lag1"]]).all():
        return None
    # Find the most-recent surface day strictly before d.
    earlier = sorted(s for s in surfaces if s < d)
    if not earlier:
        return None
    dm1 = earlier[-1]
    log_iv_tm1 = np.log(np.clip(grid_flatten(surfaces[dm1]), eps, None))
    cond = np.concatenate([
        [float(row_t["logret_lag1"])],
        [float(row_t["logret_lag2"])],
        [float(row_t["rv21_lag1"])],
        log_iv_tm1,
    ]).astype(np.float32)

    gen = _build_gen(state, device)
    cond_t = torch.from_numpy(cond).unsqueeze(0).repeat(n_samples, 1).to(device)
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(n_samples, state["config"]["noise_dim"], device=device)
    with torch.no_grad():
        fake = gen(noise, cond_t).cpu().numpy()
    # Stochastic single sample by default; if n_samples > 1, average them.
    log_iv_inc = fake[:, 1:].mean(axis=0) if n_samples > 1 else fake[0, 1:]
    log_iv_t = log_iv_tm1 + log_iv_inc
    grid = grid_unflatten(np.exp(log_iv_t))
    return np.clip(grid, 0.01, 1.5)
