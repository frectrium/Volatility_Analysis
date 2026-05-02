"""Phase-1 (same-day fit) and Phase-2 (next-day prediction) metrics.

Phase-1: each pipeline's continuous σ(k, T) (or σ via ISNN+BS-invert) is
queried at the day's actual market quote points, and we RMSE/MAE vs the
market IV per quote.

Phase-2: VolGAN outputs a 12×11 IV grid; we bilinearly interpolate to the
same market points, drop those outside the grid hull, and RMSE/MAE vs the
market IV.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from shared_eval.eval_grid import in_grid_hull, interp_surface_to_points


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

MAT_BUCKETS = [
    ("short",  0.0,  60.0 / 365.0),
    ("medium", 60.0 / 365.0, 180.0 / 365.0),
    ("long",   180.0 / 365.0, 10.0),
]
MNY_BUCKETS = [
    ("OTM_put", -10.0, -0.05),
    ("ATM",     -0.05, 0.05),
    ("OTM_call", 0.05, 10.0),
]


def _bucket_key(k_val: float, T_val: float) -> tuple[str, str]:
    mat = next((m[0] for m in MAT_BUCKETS if m[1] <= T_val < m[2]), "long")
    mny = next((m[0] for m in MNY_BUCKETS if m[1] <= k_val < m[2]), "ATM")
    return mat, mny


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(errors: np.ndarray) -> dict:
    if errors.size == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
    return {
        "n": int(errors.size),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
    }


# ---------------------------------------------------------------------------
# Phase-1
# ---------------------------------------------------------------------------

def phase1_metrics(
    quotes_dict: dict,
    eval_dates: Iterable,
    eval_at_points_fn: Callable[[pd.Timestamp, pd.DataFrame], np.ndarray],
    *,
    pipeline_name: str = "",
    verbose: bool = True,
) -> dict:
    """Compute Phase-1 metrics for a single pipeline.

    Args:
        quotes_dict: unified `quotes_dict[date] = DataFrame`.
        eval_dates: TEST dates to score on.
        eval_at_points_fn: callable(date, df_day) -> sigma_pred (np.ndarray).

    Returns a dict with overall + per-bucket + per-day metrics.
    """
    all_err: list[np.ndarray] = []
    bucket_err: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    per_day = []

    for d in eval_dates:
        d = pd.Timestamp(d).normalize()
        if d not in quotes_dict:
            continue
        df = quotes_dict[d]
        sigma_market = df["sigma_market"].to_numpy(dtype=np.float64)
        try:
            sigma_pred = eval_at_points_fn(d, df)
        except Exception as exc:
            if verbose:
                print(f"  [{pipeline_name}] eval_at_points failed on {d.date()}: {exc}")
            continue
        sigma_pred = np.asarray(sigma_pred, dtype=np.float64)
        ok = np.isfinite(sigma_pred) & np.isfinite(sigma_market)
        if not ok.any():
            continue
        err = sigma_pred[ok] - sigma_market[ok]
        all_err.append(err)
        per_day.append({
            "date": str(d.date()),
            "n": int(ok.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
        })
        k = df["log_moneyness"].to_numpy()[ok]
        T = df["T"].to_numpy()[ok]
        for i in range(err.size):
            key = _bucket_key(float(k[i]), float(T[i]))
            bucket_err[key].append(err[i:i + 1])

    flat = np.concatenate(all_err) if all_err else np.zeros(0)
    overall = _aggregate(flat)
    buckets = {
        f"{m}__{n}": _aggregate(np.concatenate(v) if v else np.zeros(0))
        for m, n in [(mb[0], nb[0]) for mb in MAT_BUCKETS for nb in MNY_BUCKETS]
        for (mm, nn), v in [((m, n), bucket_err.get((m, n), []))]
    }
    return {"overall": overall, "buckets": buckets, "per_day": per_day}


# ---------------------------------------------------------------------------
# Phase-1 (price domain)
# ---------------------------------------------------------------------------

def phase1_price_metrics(
    quotes_dict: dict,
    eval_dates: Iterable,
    eval_price_fn: Callable[[pd.Timestamp, pd.DataFrame], np.ndarray],
    *,
    pipeline_name: str = "",
    verbose: bool = True,
) -> dict:
    """Phase-1 in PRICE space.

    For A and C: native call price from the model. For B and D: BS-price the
    predicted IV. Target is df_day['mid'] (the unified mid-price).

    Returns: overall + per-bucket + per-day metrics in *raw* price units (USD).
    The caller is expected to also report relative price error if useful.
    """
    all_err: list[np.ndarray] = []
    all_rel: list[np.ndarray] = []
    bucket_err: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    per_day = []

    for d in eval_dates:
        d = pd.Timestamp(d).normalize()
        if d not in quotes_dict:
            continue
        df = quotes_dict[d]
        price_market = df["mid"].to_numpy(dtype=np.float64)
        try:
            price_pred = eval_price_fn(d, df)
        except Exception as exc:
            if verbose:
                print(f"  [{pipeline_name}] eval_price failed on {d.date()}: {exc}")
            continue
        price_pred = np.asarray(price_pred, dtype=np.float64)
        ok = np.isfinite(price_pred) & np.isfinite(price_market)
        if not ok.any():
            continue
        err = price_pred[ok] - price_market[ok]
        rel = err / np.maximum(np.abs(price_market[ok]), 1e-6)
        all_err.append(err)
        all_rel.append(rel)
        per_day.append({
            "date": str(d.date()),
            "n": int(ok.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "rel_mae": float(np.mean(np.abs(rel))),
        })
        k = df["log_moneyness"].to_numpy()[ok]
        T = df["T"].to_numpy()[ok]
        for i in range(err.size):
            key = _bucket_key(float(k[i]), float(T[i]))
            bucket_err[key].append(err[i:i + 1])

    flat = np.concatenate(all_err) if all_err else np.zeros(0)
    rel_flat = np.concatenate(all_rel) if all_rel else np.zeros(0)
    overall = _aggregate(flat)
    overall["rel_mae"] = float(np.mean(np.abs(rel_flat))) if rel_flat.size else float("nan")
    buckets = {
        f"{m}__{n}": _aggregate(np.concatenate(v) if v else np.zeros(0))
        for m, n in [(mb[0], nb[0]) for mb in MAT_BUCKETS for nb in MNY_BUCKETS]
        for (mm, nn), v in [((m, n), bucket_err.get((m, n), []))]
    }
    return {"overall": overall, "buckets": buckets, "per_day": per_day}


# ---------------------------------------------------------------------------
# Phase-2
# ---------------------------------------------------------------------------

def phase2_metrics(
    quotes_dict: dict,
    eval_dates: Iterable,
    pred_grid_fn: Callable[[pd.Timestamp], np.ndarray],
    persist_grid_fn: Callable[[pd.Timestamp], np.ndarray],
    *,
    pipeline_name: str = "",
    verbose: bool = True,
) -> dict:
    """Score VolGAN's next-day grid for a single pipeline.

    pred_grid_fn(d): returns the predicted (12,11) grid for day d, or None.
    persist_grid_fn(d): returns the persistence baseline (= the pipeline's own
        day-(d-1) grid). Used to compute Δ over persistence.
    """
    all_pred_err: list[np.ndarray] = []
    all_persist_err: list[np.ndarray] = []
    bucket_err: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    per_day = []

    for d in eval_dates:
        d = pd.Timestamp(d).normalize()
        if d not in quotes_dict:
            continue
        df = quotes_dict[d]
        k = df["log_moneyness"].to_numpy(dtype=np.float64)
        T = df["T"].to_numpy(dtype=np.float64)
        sigma_market = df["sigma_market"].to_numpy(dtype=np.float64)

        try:
            grid_pred = pred_grid_fn(d)
            grid_persist = persist_grid_fn(d)
        except Exception as exc:
            if verbose:
                print(f"  [{pipeline_name}] pred/persist failed {d.date()}: {exc}")
            continue
        if grid_pred is None or grid_persist is None:
            continue

        in_hull = in_grid_hull(k, T)
        if not in_hull.any():
            continue
        sigma_pred = interp_surface_to_points(grid_pred, k[in_hull], T[in_hull])
        sigma_persist = interp_surface_to_points(grid_persist, k[in_hull], T[in_hull])
        market = sigma_market[in_hull]
        ok = np.isfinite(sigma_pred) & np.isfinite(sigma_persist) & np.isfinite(market)
        if not ok.any():
            continue

        err_p = sigma_pred[ok] - market[ok]
        err_b = sigma_persist[ok] - market[ok]
        all_pred_err.append(err_p)
        all_persist_err.append(err_b)
        per_day.append({
            "date": str(d.date()),
            "n": int(ok.sum()),
            "rmse_pred": float(np.sqrt(np.mean(err_p ** 2))),
            "rmse_persist": float(np.sqrt(np.mean(err_b ** 2))),
            "mae_pred": float(np.mean(np.abs(err_p))),
        })
        k_ok = k[in_hull][ok]
        T_ok = T[in_hull][ok]
        for i in range(err_p.size):
            key = _bucket_key(float(k_ok[i]), float(T_ok[i]))
            bucket_err[key].append(err_p[i:i + 1])

    flat = np.concatenate(all_pred_err) if all_pred_err else np.zeros(0)
    persist_flat = np.concatenate(all_persist_err) if all_persist_err else np.zeros(0)
    overall = _aggregate(flat)
    persist = _aggregate(persist_flat)
    delta = (
        (persist["rmse"] - overall["rmse"]) / persist["rmse"]
        if persist["rmse"] and np.isfinite(persist["rmse"]) and persist["rmse"] > 0
        else float("nan")
    )
    buckets = {
        f"{m}__{n}": _aggregate(np.concatenate(v) if v else np.zeros(0))
        for m, n in [(mb[0], nb[0]) for mb in MAT_BUCKETS for nb in MNY_BUCKETS]
        for (mm, nn), v in [((m, n), bucket_err.get((m, n), []))]
    }
    return {
        "overall": overall,
        "persistence": persist,
        "delta_over_persistence": delta,
        "buckets": buckets,
        "per_day": per_day,
    }
