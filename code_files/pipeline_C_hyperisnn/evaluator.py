"""Evaluation, constraint checks, inference-speed measurement, and sample
prediction tables.

All metrics are reported in dollar terms (price RMSE/MAE) AND in IV terms
(after Newton inversion of predicted prices). Bucketed reports by maturity
and moneyness.
"""

import time
from typing import Callable, Dict, List

import numpy as np
import torch

from .bs_utils import bs_call, implied_vol
from .config import HyperConfig
from .model import HyperISNN
from .trainer import collate, sample_episode, to_device


# ----------------------------------------------------------------------------
# Whole-dataset evaluation
# ----------------------------------------------------------------------------

def evaluate(
    model: HyperISNN,
    dataset: List[dict],
    cfg_hyper: HyperConfig,
    device: torch.device,
    name: str = "test",
    log_fn: Callable[[str], None] = print,
    chunk_size: int = 32,
) -> Dict[str, float]:
    log_fn(f"\n--- Evaluating on {name} ({len(dataset)} days) ---")
    model.eval()

    chunks_pred, chunks_true = [], []
    chunks_K, chunks_T, chunks_iv, chunks_vega, chunks_S, chunks_r = [], [], [], [], [], []

    rng = np.random.default_rng(0)        # deterministic eval splits

    with torch.no_grad():
        for j in range(0, len(dataset), chunk_size):
            chunk = dataset[j:j + chunk_size]
            eps = [(d, *sample_episode(d, cfg_hyper, rng)) for d in chunk]
            b = to_device(collate(eps, cfg_hyper), device)
            pred = model(b["q"], b["mask"], b["ctx"], b["m"], b["T"], b["r"]).cpu().numpy().squeeze(-1)
            tmask = b["tmask"].cpu().numpy().squeeze(-1).astype(bool)
            true_ = b["C"].cpu().numpy().squeeze(-1)
            S_arr = b["S"].cpu().numpy().squeeze(-1).squeeze(-1)
            for i in range(pred.shape[0]):
                m = tmask[i]
                S = S_arr[i]
                day = chunk[i]
                t_idx = eps[i][2]
                chunks_pred.append(pred[i, m] * S)
                chunks_true.append(true_[i, m] * S)
                chunks_K.append(day["K"][t_idx])
                chunks_T.append(day["T"][t_idx])
                chunks_iv.append(day["iv"][t_idx])
                chunks_vega.append(day["vega"][t_idx])
                chunks_S.append(np.full(int(m.sum()), S, dtype=np.float32))
                chunks_r.append(np.full(int(m.sum()), day["r"], dtype=np.float32))

    pred_d = np.concatenate(chunks_pred)
    true_d = np.concatenate(chunks_true)
    K_d    = np.concatenate(chunks_K)
    T_d    = np.concatenate(chunks_T)
    iv_true = np.concatenate(chunks_iv)
    vg_d   = np.concatenate(chunks_vega)
    S_d    = np.concatenate(chunks_S)
    r_d    = np.concatenate(chunks_r)

    err = pred_d - true_d
    rmse_d = float(np.sqrt(np.mean(err ** 2)))
    mae_d  = float(np.mean(np.abs(err)))
    rel_d  = float(np.mean(np.abs(err) / np.maximum(true_d, 0.5)))

    # IV inversion of predicted prices (per (S, r))
    iv_pred = np.full_like(pred_d, np.nan)
    for S_v in np.unique(S_d):
        msk = S_d == S_v
        if msk.sum() == 0:
            continue
        for r_v in np.unique(r_d[msk]):
            mm = msk & (r_d == r_v)
            iv_pred[mm] = implied_vol(pred_d[mm], float(S_v), K_d[mm], T_d[mm], float(r_v))
    valid = (~np.isnan(iv_pred)) & (~np.isnan(iv_true)) & (vg_d > 1e-2)
    iv_err = iv_pred[valid] - iv_true[valid]
    iv_rmse = float(np.sqrt(np.mean(iv_err ** 2))) if valid.sum() else float("nan")
    iv_mae  = float(np.mean(np.abs(iv_err))) if valid.sum() else float("nan")
    iv_valid_pct = 100.0 * valid.sum() / max(len(valid), 1)

    log_fn(f"  N evaluated contracts: {len(pred_d)}")
    log_fn(f"  Price ($)           RMSE {rmse_d:8.4f}   MAE {mae_d:8.4f}   relMAE {rel_d:7.4f}")
    log_fn(f"  IV (proper Newton)  RMSE {iv_rmse:.5f}   MAE {iv_mae:.5f}   "
           f"valid {iv_valid_pct:.1f}% (filtered by vega > 0.01)")

    log_fn("  By maturity bucket:")
    for lo, hi, lbl in [(0, 30, "  0– 30d"), (30, 90, " 30– 90d"), (90, 365, " 90–365d")]:
        m = ((T_d * 365 >= lo) & (T_d * 365 < hi))
        if m.sum() > 0:
            r1 = float(np.sqrt((err[m] ** 2).mean()))
            mv = m & valid
            r2 = float(np.abs(iv_pred[mv] - iv_true[mv]).mean()) if mv.sum() else float("nan")
            log_fn(f"    {lbl}: N={m.sum():6d}  price RMSE ${r1:7.4f}   IV MAE {r2:.5f}")

    log_fn("  By moneyness bucket (K/S):")
    mny = K_d / S_d
    for lo, hi, lbl in [
        (0.7,  0.95, "deep ITM   (0.70–0.95)"),
        (0.95, 1.05, "ATM        (0.95–1.05)"),
        (1.05, 1.30, "OTM        (1.05–1.30)"),
    ]:
        m = (mny >= lo) & (mny < hi)
        if m.sum() > 0:
            r1 = float(np.sqrt((err[m] ** 2).mean()))
            mv = m & valid
            r2 = float(np.abs(iv_pred[mv] - iv_true[mv]).mean()) if mv.sum() else float("nan")
            log_fn(f"    {lbl}: N={m.sum():6d}  price RMSE ${r1:7.4f}   IV MAE {r2:.5f}")

    return dict(
        price_rmse=rmse_d, price_mae=mae_d,
        iv_rmse=iv_rmse, iv_mae=iv_mae,
        n=len(pred_d),
    )


# ----------------------------------------------------------------------------
# Constraint check (architectural — should pass 100%)
# ----------------------------------------------------------------------------

def check_constraints(
    model: HyperISNN,
    dataset: List[dict],
    cfg_hyper: HyperConfig,
    device: torch.device,
    n_days: int = 20,
    log_fn: Callable[[str], None] = print,
):
    log_fn(f"\n--- Constraint check on {n_days} held-out days ---")
    model.eval()

    fail_K_mono = fail_K_conv = fail_T_mono = 0
    nontrivial_K = nontrivial_T = 0
    rng = np.random.default_rng(123)

    with torch.no_grad():
        step = max(1, len(dataset) // n_days)
        for trial in range(n_days):
            day = dataset[(trial * step) % len(dataset)]
            c_idx, _ = sample_episode(day, cfg_hyper, rng)
            t_idx = np.array([0])
            b = to_device(collate([(day, c_idx, t_idx)], cfg_hyper), device)

            # Strike sweep at fixed T
            mg = torch.linspace(0.7, 1.3, 300, device=device).view(-1, 1)
            Tg_fixed = torch.full_like(mg, 0.25)
            r_fixed = torch.full_like(mg, float(day["r"]))
            Cm = model.predict_surface(b["q"], b["mask"], b["ctx"], mg, Tg_fixed, r_fixed).squeeze(-1)

            # Maturity sweep at fixed m=1.0
            Tg = torch.linspace(0.02, 1.0, 200, device=device).view(-1, 1)
            mg_fixed = torch.full_like(Tg, 1.0)
            r_T = torch.full_like(Tg, float(day["r"]))
            CT = model.predict_surface(b["q"], b["mask"], b["ctx"], mg_fixed, Tg, r_T).squeeze(-1)

            dCm  = torch.diff(Cm)
            d2Cm = torch.diff(Cm, n=2)
            dCT  = torch.diff(CT)

            if not (dCm <= 1e-6).all().item():
                fail_K_mono += 1
            if not (d2Cm >= -1e-6).all().item():
                fail_K_conv += 1
            if not (dCT >= -1e-6).all().item():
                fail_T_mono += 1
            if (Cm.max() - Cm.min()).item() > 1e-5:
                nontrivial_K += 1
            if (CT.max() - CT.min()).item() > 1e-5:
                nontrivial_T += 1

    log_fn(f"  monotone-↓ in K  : {n_days - fail_K_mono}/{n_days}")
    log_fn(f"  convex     in K  : {n_days - fail_K_conv}/{n_days}")
    log_fn(f"  monotone-↑ in T  : {n_days - fail_T_mono}/{n_days}")
    log_fn(f"  non-trivial in K : {nontrivial_K}/{n_days}  (surface varies in strike)")
    log_fn(f"  non-trivial in T : {nontrivial_T}/{n_days}  (surface varies in maturity)")


# ----------------------------------------------------------------------------
# Inference-speed measurement
# ----------------------------------------------------------------------------

def speed_test(
    model: HyperISNN,
    dataset: List[dict],
    cfg_hyper: HyperConfig,
    device: torch.device,
    n_grid: int = 10000,
    log_fn: Callable[[str], None] = print,
):
    log_fn("\n--- Inference speed (single forward) ---")
    model.eval()
    day = dataset[0]
    rng = np.random.default_rng(0)
    c_idx, _ = sample_episode(day, cfg_hyper, rng)
    b = to_device(collate([(day, c_idx, np.array([0]))], cfg_hyper), device)

    with torch.no_grad():
        # Warm up
        for _ in range(5):
            flat = model.generate_flat_weights(b["q"], b["mask"], b["ctx"])
        # Time hypernet
        N = 500
        t0 = time.time()
        for _ in range(N):
            flat = model.generate_flat_weights(b["q"], b["mask"], b["ctx"])
        t_g = (time.time() - t0) / N * 1000

        # Time ISNN on a 10k point grid
        mg = torch.rand(n_grid, 1, device=device) * 0.6 + 0.7
        Tg = torch.rand(n_grid, 1, device=device) * 0.95 + 0.05
        rg = torch.full_like(mg, float(day["r"]))
        for _ in range(5):
            model.predict_surface(b["q"], b["mask"], b["ctx"], mg, Tg, rg)
        N2 = 100
        t0 = time.time()
        for _ in range(N2):
            model.predict_surface(b["q"], b["mask"], b["ctx"], mg, Tg, rg)
        t_h = (time.time() - t0) / N2 * 1000

    log_fn(f"  hypernet g forward             : {t_g:6.2f} ms")
    log_fn(f"  ISNN h on {n_grid:,} (m,T) points : {t_h:6.2f} ms")
    log_fn(f"  Total to build a {n_grid:,} surface : {t_g + t_h:6.2f} ms")


# ----------------------------------------------------------------------------
# Sample predictions for visual inspection
# ----------------------------------------------------------------------------

def sample_predictions(
    model: HyperISNN,
    dataset: List[dict],
    cfg_hyper: HyperConfig,
    device: torch.device,
    n_show: int = 10,
    log_fn: Callable[[str], None] = print,
):
    log_fn("\n--- Sample predictions on a random test day ---")
    day = dataset[len(dataset) // 2]
    log_fn(f"  Date: {day['date'].date()}   S = {day['S']:.2f}   r = {day['r']:.4f}")

    rng = np.random.default_rng(7)
    c_idx, t_idx = sample_episode(day, cfg_hyper, rng)
    t_idx = t_idx[:n_show]

    log_fn(f"  Context (fed to g): {len(c_idx)} contracts")
    log_fn(f"  {'  K':>10}  {'T(d)':>6}  {'mid$':>10}  {'IV':>7}")
    for i in c_idx[:6]:
        log_fn(f"  {day['K'][i]:10.2f}  {int(day['T'][i]*365):6d}  "
               f"{day['price'][i]:10.4f}  {day['iv'][i]:7.4f}")
    if len(c_idx) > 6:
        log_fn(f"  ... ({len(c_idx)-6} more)")

    b = to_device(collate([(day, c_idx, t_idx)], cfg_hyper), device)
    with torch.no_grad():
        pred = model(b["q"], b["mask"], b["ctx"], b["m"], b["T"], b["r"]).cpu().numpy().squeeze()
    pred_d = pred * day["S"]
    if pred_d.ndim == 0:
        pred_d = pred_d.reshape(1)
    true_d = day["price"][t_idx]
    iv_true = day["iv"][t_idx]
    iv_pred = implied_vol(pred_d, day["S"], day["K"][t_idx], day["T"][t_idx], day["r"])

    log_fn("  Held-out targets:")
    log_fn(f"  {'  K':>10}  {'T(d)':>6}  {'true$':>10}  {'pred$':>10}  "
           f"{'err$':>10}  {'true IV':>8}  {'pred IV':>8}")
    for j in range(min(n_show, len(t_idx))):
        i = t_idx[j]
        log_fn(f"  {day['K'][i]:10.2f}  {int(day['T'][i]*365):6d}  "
               f"{true_d[j]:10.4f}  {pred_d[j]:10.4f}  {pred_d[j]-true_d[j]:+10.4f}  "
               f"{iv_true[j]:8.4f}  {iv_pred[j]:8.4f}")
