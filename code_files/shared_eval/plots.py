"""Comparison plots (the seven families described in plan.md §7).

Reads the JSONs at `data/unified/results/{phase1_fit,phase2_predict}.json` and
the surface caches under `data/unified/surfaces/`. Writes PNGs under
`plots/`.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared_eval.eval_grid import (
    LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID, GRID_SHAPE, interp_surface_to_points,
)
from scipy.stats import norm

def _black76_call_vec(F, K, T, sigma, disc):
    F = np.asarray(F)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)
    disc = np.asarray(disc)
    
    sigT = sigma * np.sqrt(T)
    d1 = (np.log(np.maximum(F, 1e-12) / np.maximum(K, 1e-12)) + 0.5 * sigma * sigma * T) / sigT
    d2 = d1 - sigT
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


PIPELINES = ("A", "B", "C", "D")
COLOURS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(paths: dict) -> dict:
    p1_path = paths["results"] / "phase1_fit.json"
    p2_path = paths["results"] / "phase2_predict.json"
    p1p_path = paths["results"] / "phase1_price.json"
    p1 = json.loads(p1_path.read_text()) if p1_path.exists() else {}
    p2 = json.loads(p2_path.read_text()) if p2_path.exists() else {}
    p1_price = json.loads(p1p_path.read_text()) if p1p_path.exists() else {}
    surfaces = {}
    for name in PIPELINES:
        f = paths["surfaces"] / f"{name}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                surfaces[name] = pickle.load(fh)
    return {"phase1": p1, "phase1_price": p1_price, "phase2": p2, "surfaces": surfaces}


# ---------------------------------------------------------------------------
# 1. Headline bar chart
# ---------------------------------------------------------------------------

def plot_headline(p1, p2, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(PIPELINES))
    w = 0.4
    p1_vals = [p1.get(n, {}).get("overall", {}).get("rmse", np.nan) for n in PIPELINES]
    p2_vals = [p2.get(n, {}).get("overall", {}).get("rmse", np.nan) for n in PIPELINES]
    ax.bar(x - w/2, p1_vals, w, label="Phase-1 (same-day fit)", color="steelblue")
    ax.bar(x + w/2, p2_vals, w, label="Phase-2 (next-day predict)", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(PIPELINES)
    ax.set_ylabel("IV RMSE")
    ax.set_title("Phase-1 vs Phase-2 RMSE per pipeline")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Phase-1 vs Phase-2 scatter
# ---------------------------------------------------------------------------

def plot_p1_p2_scatter(p1, p2, out: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for n in PIPELINES:
        x = p1.get(n, {}).get("overall", {}).get("rmse", np.nan)
        y = p2.get(n, {}).get("overall", {}).get("rmse", np.nan)
        ax.scatter(x, y, s=80, c=COLOURS[n], label=n)
        ax.annotate(n, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Phase-1 (same-day) RMSE")
    ax.set_ylabel("Phase-2 (next-day) RMSE")
    ax.set_title("Same-day fit vs next-day prediction")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Per-bucket heatmap
# ---------------------------------------------------------------------------

def _bucket_matrix(metrics: dict, key: str = "rmse") -> tuple[np.ndarray, list, list]:
    mats = ["short", "medium", "long"]
    mns = ["OTM_put", "ATM", "OTM_call"]
    rows = []
    for n in PIPELINES:
        b = metrics.get(n, {}).get("buckets", {})
        row = [b.get(f"{m}__{x}", {}).get(key, np.nan) for m in mats for x in mns]
        rows.append(row)
    return np.asarray(rows), mats, mns


def plot_bucket_heatmap(metrics, out: Path, title: str):
    M, mats, mns = _bucket_matrix(metrics, "rmse")
    if not np.isfinite(M).any():
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(PIPELINES))); ax.set_yticklabels(PIPELINES)
    labels = [f"{m}\n{x}" for m in mats for x in mns]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Slice plots on 3 chosen TEST dates (smile + term-structure)
# ---------------------------------------------------------------------------

def _pick_test_dates(p1: dict) -> list:
    """Pick 3 TEST dates (calm/trending/spike) from per-day RMSE distribution."""
    days = p1.get("A", {}).get("per_day", [])
    if not days:
        return []
    rmse = np.array([d["rmse"] for d in days])
    if rmse.size < 3:
        return [d["date"] for d in days]
    idx_low = int(np.argmin(rmse))
    idx_hi = int(np.argmax(rmse))
    idx_mid = int(np.argsort(rmse)[len(rmse) // 2])
    return [days[idx_low]["date"], days[idx_mid]["date"], days[idx_hi]["date"]]


def plot_slices(surfaces: dict, payload: dict, dates: list, out_dir: Path):
    if not dates:
        return
    quotes = payload["quotes_dict"]
    for d_str in dates:
        d = pd.Timestamp(d_str).normalize()
        if d not in quotes:
            continue
        df = quotes[d]
        # 4a: smile slice at tau ≈ 30d
        target_T = 30 / 365.0
        df_slice = df[(df["maturity_days"] >= 20) & (df["maturity_days"] <= 45)]
        if len(df_slice) > 5:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(df_slice["log_moneyness"], df_slice["sigma_market"],
                       c="black", s=15, label="market", alpha=0.6)
            kg = np.linspace(df["log_moneyness"].min(), df["log_moneyness"].max(), 50)
            for n in PIPELINES:
                if n not in surfaces or d not in surfaces[n]:
                    continue
                surf = surfaces[n][d]
                ivs = interp_surface_to_points(surf, kg, np.full_like(kg, target_T))
                ax.plot(kg, ivs, color=COLOURS[n], label=f"{n}", lw=1.5)
            ax.set_xlabel("log-moneyness k = log(K/F)")
            ax.set_ylabel("IV")
            ax.set_title(f"Smile @ τ≈30d  ({d.date()})")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(out_dir / f"slice_smile_{d.date()}.png", dpi=150)
            plt.close(fig)

def plot_price_slices(surfaces: dict, payload: dict, dates: list, out_dir: Path):
    if not dates:
        return
    quotes = payload["quotes_dict"]
    for d_str in dates:
        d = pd.Timestamp(d_str).normalize()
        if d not in quotes:
            continue
        df = quotes[d]
        target_T = 30 / 365.0
        df_slice = df[(df["maturity_days"] >= 20) & (df["maturity_days"] <= 45)]
        if len(df_slice) > 5:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(df_slice["log_moneyness"], df_slice["mid"],
                       c="black", s=15, label="market price", alpha=0.6)
            kg = np.linspace(df_slice["log_moneyness"].min(), df_slice["log_moneyness"].max(), 50)
            
            # Use mean F and discount for the slice to convert continuous curves
            mean_F = df_slice["fwd_price"].mean()
            mean_disc = df_slice["discount"].mean()
            K_grid = mean_F * np.exp(kg)
            
            for n in PIPELINES:
                if n not in surfaces or d not in surfaces[n]:
                    continue
                surf = surfaces[n][d]
                ivs = interp_surface_to_points(surf, kg, np.full_like(kg, target_T))
                # Convert IV to Price
                prices = _black76_call_vec(mean_F, K_grid, target_T, ivs, mean_disc)
                ax.plot(kg, prices, color=COLOURS[n], label=f"{n}", lw=1.5)
                
            ax.set_xlabel("log-moneyness k = log(K/F)")
            ax.set_ylabel("Price ($)")
            ax.set_title(f"Price Surface @ τ≈30d  ({d.date()})")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(out_dir / f"slice_price_{d.date()}.png", dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Rolling RMSE time series on TEST
# ---------------------------------------------------------------------------

def plot_rolling(p1: dict, p2: dict, out: Path, window: int = 20):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, metrics, title in [(axes[0], p1, "Phase-1 (same-day) rolling RMSE"),
                                (axes[1], p2, "Phase-2 (next-day) rolling RMSE")]:
        for n in PIPELINES:
            days = metrics.get(n, {}).get("per_day", [])
            if not days:
                continue
            df = pd.DataFrame(days)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            key = "rmse" if "rmse" in df.columns else "rmse_pred"
            roll = df[key].rolling(window).mean()
            ax.plot(df["date"], roll, label=n, color=COLOURS[n])
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Price-domain plots (Phase-1 native price for A/C, BS-priced IV for B/D)
# ---------------------------------------------------------------------------

def plot_price_headline(p1_price: dict, out: Path):
    """Per-pipeline price RMSE and relative-MAE bars."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(PIPELINES))
    rmse = [p1_price.get(n, {}).get("overall", {}).get("rmse", np.nan) for n in PIPELINES]
    rel  = [p1_price.get(n, {}).get("overall", {}).get("rel_mae", np.nan) for n in PIPELINES]
    cols = [COLOURS[n] for n in PIPELINES]
    axes[0].bar(x, rmse, color=cols)
    axes[0].set_xticks(x); axes[0].set_xticklabels(PIPELINES)
    axes[0].set_ylabel("Price RMSE  ($)")
    axes[0].set_title("Phase-1 price-domain RMSE")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x, rel, color=cols)
    axes[1].set_xticks(x); axes[1].set_xticklabels(PIPELINES)
    axes[1].set_ylabel("Relative price MAE")
    axes[1].set_title("Phase-1 relative price MAE")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_price_iv_AC(p1: dict, p1_price: dict, out: Path):
    """A vs C in IV space and price space — does C catch up to A in price?"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pipes = ("A", "C")

    iv_rmse = [p1.get(n, {}).get("overall", {}).get("rmse", np.nan) for n in pipes]
    pr_rmse = [p1_price.get(n, {}).get("overall", {}).get("rmse", np.nan) for n in pipes]
    pr_rel  = [p1_price.get(n, {}).get("overall", {}).get("rel_mae", np.nan) for n in pipes]

    cols = [COLOURS[n] for n in pipes]
    x = np.arange(len(pipes))
    axes[0].bar(x, iv_rmse, color=cols)
    axes[0].set_xticks(x); axes[0].set_xticklabels(pipes)
    axes[0].set_title("IV-domain RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, pr_rmse, color=cols)
    axes[1].set_xticks(x); axes[1].set_xticklabels(pipes)
    axes[1].set_title("Price-domain RMSE  ($)")
    axes[1].set_ylabel("Price RMSE")
    for i, v in enumerate(pr_rel):
        axes[1].annotate(f"rel-MAE {v:.3f}", (i, pr_rmse[i]),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("A vs C — IV-domain vs price-domain  (Phase-1)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def render_all(paths: dict, payload: dict, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load(paths)
    p1 = bundle["phase1"]
    p2 = bundle["phase2"]
    p1_price = bundle["phase1_price"]
    surfaces = bundle["surfaces"]

    if p1 and p2:
        plot_headline(p1, p2, out_dir / "01_headline_rmse.png")
        plot_p1_p2_scatter(p1, p2, out_dir / "02_p1_vs_p2_scatter.png")
    if p1:
        plot_bucket_heatmap(p1, out_dir / "03a_buckets_phase1.png",
                            "Phase-1 RMSE by bucket")
    if p2:
        plot_bucket_heatmap(p2, out_dir / "03b_buckets_phase2.png",
                            "Phase-2 RMSE by bucket")
    if p1 and surfaces:
        test_dates = _pick_test_dates(p1)
        plot_slices(surfaces, payload, test_dates, out_dir)
        plot_price_slices(surfaces, payload, test_dates, out_dir)
    if p1 and p2:
        plot_rolling(p1, p2, out_dir / "06_rolling_rmse.png")
    if p1_price:
        plot_price_headline(p1_price, out_dir / "07a_price_headline.png")
    if p1 and p1_price:
        plot_price_iv_AC(p1, p1_price, out_dir / "07b_price_iv_AC.png")
    
    try:
        from shared_eval.add_plots import add_new_plots
        add_new_plots(surfaces, p1_price, payload, out_dir)
    except Exception as e:
        print(f"Error adding new plots: {e}")
        
    return out_dir
