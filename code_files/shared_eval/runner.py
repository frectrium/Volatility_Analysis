"""Cross-pipeline runner.

Orchestrates the full unified comparison:
  1. Load (or build) the unified dataset.
  2. For each pipeline P in {A, B, C, D}: train (or reload) and dump
     `surfaces[P][date] = (12, 11) ndarray`.
  3. Phase-1 metrics: each pipeline's continuous σ scored at TEST market quotes.
  4. For each pipeline P: train a VolGAN on P's TRAIN surfaces.
  5. Phase-2 metrics: VolGAN's next-day grid scored on TEST.
  6. Save phase1/phase2 JSON + Markdown summary tables.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code_files") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code_files"))

from shared_eval.unified_dataset import load_unified_dataset  # noqa: E402
from shared_eval.metrics import phase1_metrics, phase1_price_metrics, phase2_metrics  # noqa: E402
from shared_eval.adapters import (  # noqa: E402
    pipeline_a_isnn as P_A,
    pipeline_b_hyperiv as P_B,
    pipeline_c_hyperisnn as P_C,
    pipeline_d_ssvi as P_D,
)
from shared_eval import volgan_adapter as VG  # noqa: E402


PIPELINES = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def make_paths(quick: bool, root: Path = PROJECT_ROOT) -> dict:
    base = root / "data" / "unified" / ("quick" if quick else ".")
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return {
        "root": base,
        "models": base / "models",
        "surfaces": base / "surfaces",
        "volgan": base / "volgan_ckpts",
        "results": base / "results",
    }


# ---------------------------------------------------------------------------
# Surface generation
# ---------------------------------------------------------------------------

def _all_dates(splits) -> list:
    return sorted(splits["train"]) + sorted(splits["val"]) + sorted(splits["test"])


def train_or_load_pipeline(
    name: str, payload: dict, *, quick: bool, paths: dict, rebuild: bool, verbose: bool
):
    quotes = payload["quotes_dict"]
    splits = payload["splits"]

    if name == "A":
        cache = paths["models"] / "A_isnn.pkl"
        if cache.exists() and not rebuild:
            return P_A.load(cache)
        return P_A.train_all(quotes, _all_dates(splits), quick=quick,
                             cache_path=cache, verbose=verbose)
    if name == "D":
        cache = paths["models"] / "D_ssvi.pkl"
        if cache.exists() and not rebuild:
            return P_D.load(cache)
        return P_D.train_all(quotes, _all_dates(splits), quick=quick,
                             cache_path=cache, verbose=verbose)
    if name == "B":
        cache_dir = paths["models"] / "B_hyperiv"
        if (cache_dir / "adapter_state.pkl").exists() and not rebuild:
            state = P_B.load(cache_dir)
        else:
            state = P_B.train(
                quotes, splits["train"], splits["val"],
                quick=quick, cache_dir=cache_dir, verbose=verbose,
            )
        omega = P_B.precompute_omega(state, quotes, _all_dates(splits))
        return {"state": state, "omega": omega}
    if name == "C":
        cache_dir = paths["models"] / "C_hyperisnn"
        if (cache_dir / "adapter_state.pkl").exists() and not rebuild:
            state = P_C.load(cache_dir)
        else:
            state = P_C.train(
                payload, splits["train"], splits["val"],
                quick=quick, cache_dir=cache_dir, verbose=verbose,
            )
        days = P_C.precompute_days(state, payload, _all_dates(splits))
        return {"state": state, "days": days}
    raise ValueError(name)


def materialise_surfaces(name: str, model_state, payload: dict, *, paths: dict,
                          verbose: bool, rebuild: bool = False) -> dict:
    """Return {date: (12, 11) grid} for every date in TRAIN ∪ VAL ∪ TEST."""
    cache = paths["surfaces"] / f"{name}.pkl"
    if cache.exists() and not rebuild:
        with open(cache, "rb") as f:
            return pickle.load(f)
    splits = payload["splits"]
    dates = _all_dates(splits)
    out: dict = {}
    for i, d in enumerate(dates):
        d = pd.Timestamp(d).normalize()
        if name == "A":
            g = P_A.eval_grid(model_state, d)
        elif name == "B":
            g = P_B.eval_grid(model_state["state"], model_state["omega"], d)
        elif name == "C":
            g = P_C.eval_grid(model_state["state"], model_state["days"], d)
        elif name == "D":
            g = P_D.eval_grid(model_state, d)
        else:
            raise ValueError(name)
        if g is not None and np.isfinite(g).all():
            out[d] = g.astype(np.float32)
        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"  [{name}] surfaces {i+1}/{len(dates)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  [{name}] saved surfaces to {cache} ({len(out)} dates)")
    return out


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def run_phase1(model_states: dict, payload: dict, *, paths: dict, verbose: bool) -> dict:
    quotes = payload["quotes_dict"]
    test_dates = payload["splits"]["test"]
    out = {}
    for name in PIPELINES:
        if name not in model_states:
            continue
        ms = model_states[name]
        if name == "A":
            fn = lambda d, df: P_A.eval_at_points(ms, d, df)
        elif name == "B":
            fn = lambda d, df: P_B.eval_at_points(ms["state"], ms["omega"], d, df)
        elif name == "C":
            fn = lambda d, df: P_C.eval_at_points(ms["state"], ms["days"], d, df)
        elif name == "D":
            fn = lambda d, df: P_D.eval_at_points(ms, d, df)
        if verbose:
            print(f"\n[Phase-1] scoring pipeline {name} on {len(test_dates)} TEST days...")
        out[name] = phase1_metrics(quotes, test_dates, fn, pipeline_name=name, verbose=verbose)
    paths["results"].mkdir(parents=True, exist_ok=True)
    with open(paths["results"] / "phase1_fit.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def run_phase1_price(model_states: dict, payload: dict, *, paths: dict, verbose: bool) -> dict:
    """Phase-1 in PRICE space (native price for A/C; BS-priced IV for B/D)."""
    quotes = payload["quotes_dict"]
    test_dates = payload["splits"]["test"]
    out = {}
    for name in PIPELINES:
        if name not in model_states:
            continue
        ms = model_states[name]
        if name == "A":
            fn = lambda d, df: P_A.eval_price_at_points(ms, d, df)
        elif name == "B":
            fn = lambda d, df: P_B.eval_price_at_points(ms["state"], ms["omega"], d, df)
        elif name == "C":
            fn = lambda d, df: P_C.eval_price_at_points(ms["state"], ms["days"], d, df)
        elif name == "D":
            fn = lambda d, df: P_D.eval_price_at_points(ms, d, df)
        if verbose:
            print(f"\n[Phase-1 PRICE] scoring pipeline {name} on {len(test_dates)} TEST days...")
        out[name] = phase1_price_metrics(quotes, test_dates, fn, pipeline_name=name, verbose=verbose)
    paths["results"].mkdir(parents=True, exist_ok=True)
    with open(paths["results"] / "phase1_price.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def train_volgan_for(name: str, surfaces_p: dict, payload: dict, *, paths: dict,
                     quick: bool, rebuild: bool, verbose: bool) -> dict:
    cache = paths["volgan"] / f"{name}.pkl"
    if cache.exists() and not rebuild:
        return VG.load(cache)
    return VG.train(
        surfaces_p, payload["side_info"],
        payload["splits"]["train"], payload["splits"]["val"],
        quick=quick, cache_path=cache, verbose=verbose,
    )


def run_phase2(surfaces_by_pipe: dict, volgan_states: dict, payload: dict,
               *, paths: dict, verbose: bool) -> dict:
    quotes = payload["quotes_dict"]
    side = payload["side_info"]
    test_dates = payload["splits"]["test"]
    out = {}
    for name in PIPELINES:
        if name not in volgan_states:
            continue
        surf = surfaces_by_pipe[name]
        vstate = volgan_states[name]

        def pred_grid(d, _surf=surf, _vstate=vstate):
            return VG.predict_grid(_vstate, _surf, side, d)

        def persist_grid(d, _surf=surf):
            earlier = sorted(s for s in _surf if s < pd.Timestamp(d).normalize())
            return _surf[earlier[-1]] if earlier else None

        if verbose:
            print(f"\n[Phase-2] scoring pipeline {name} on {len(test_dates)} TEST days...")
        out[name] = phase2_metrics(
            quotes, test_dates, pred_grid, persist_grid,
            pipeline_name=name, verbose=verbose,
        )
    paths["results"].mkdir(parents=True, exist_ok=True)
    with open(paths["results"] / "phase2_predict.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_summary_md(phase1: dict, phase2: dict, paths: dict, *,
                     phase1_price: dict | None = None) -> Path:
    lines = ["# Unified cross-pipeline summary", ""]
    lines.append("## IV-domain (Phase-1 same-day fit, Phase-2 next-day predict)")
    lines.append("")
    lines.append("| Pipe | Phase-1 RMSE | Phase-1 MAE | Phase-2 RMSE | Phase-2 MAE | Δ over persistence |")
    lines.append("|------|--------------|-------------|--------------|-------------|---------------------|")
    for name in PIPELINES:
        p1 = phase1.get(name, {}).get("overall", {})
        p2 = phase2.get(name, {}).get("overall", {})
        delta = phase2.get(name, {}).get("delta_over_persistence", float("nan"))
        lines.append(
            f"| {name} | "
            f"{p1.get('rmse', float('nan')):.5f} | "
            f"{p1.get('mae', float('nan')):.5f} | "
            f"{p2.get('rmse', float('nan')):.5f} | "
            f"{p2.get('mae', float('nan')):.5f} | "
            f"{delta:+.3f} |"
        )

    if phase1_price:
        lines.append("")
        lines.append("## Price-domain Phase-1 (native price for A,C; BS-priced IV for B,D)")
        lines.append("")
        lines.append("| Pipe | Price RMSE ($) | Price MAE ($) | Rel-MAE | n quotes |")
        lines.append("|------|----------------|----------------|---------|----------|")
        for name in PIPELINES:
            o = phase1_price.get(name, {}).get("overall", {})
            lines.append(
                f"| {name} | "
                f"{o.get('rmse', float('nan')):.4f} | "
                f"{o.get('mae', float('nan')):.4f} | "
                f"{o.get('rel_mae', float('nan')):.4f} | "
                f"{o.get('n', 0)} |"
            )

    out = paths["results"] / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run(
    *,
    pipelines: tuple = PIPELINES,
    quick: bool = False,
    phase: int | None = None,    # None = both, 1 = phase-1 only, 2 = phase-2 only
    rebuild: bool | tuple = False,
    verbose: bool = True,
) -> dict:
    # Normalise `rebuild`:
    #   False         → use all caches
    #   True          → rebuild every selected pipeline (legacy behaviour)
    #   ("C",) etc.   → rebuild only the named pipelines (caches kept for the rest)
    if rebuild is True:
        rebuild_set = set(pipelines)
    elif rebuild in (False, None):
        rebuild_set = set()
    else:
        rebuild_set = set(rebuild)
    paths = make_paths(quick=quick)
    payload = load_unified_dataset()
    if quick:
        # Trim splits for fast verification.
        payload = {**payload, "splits": {
            "train": payload["splits"]["train"][:30],
            "val":   payload["splits"]["val"][:5],
            "test":  payload["splits"]["test"][:10],
        }}
        if verbose:
            print("[quick] using 30 train + 5 val + 10 test days")

    model_states = {}
    surfaces = {}
    for name in pipelines:
        if verbose:
            tag = " [REBUILD]" if name in rebuild_set else ""
            print(f"\n=== Pipeline {name}: train/load{tag} ===")
        ms = train_or_load_pipeline(name, payload, quick=quick, paths=paths,
                                    rebuild=(name in rebuild_set), verbose=verbose)
        model_states[name] = ms
        if verbose:
            print(f"=== Pipeline {name}: materialise (12x11) surfaces ===")
        surfaces[name] = materialise_surfaces(name, ms, payload, paths=paths,
                                              verbose=verbose,
                                              rebuild=(name in rebuild_set))

    phase1 = phase1_price = phase2 = {}
    if phase in (None, 1):
        phase1 = run_phase1(model_states, payload, paths=paths, verbose=verbose)
        phase1_price = run_phase1_price(model_states, payload, paths=paths, verbose=verbose)
    if phase in (None, 2):
        volgan_states = {}
        for name in pipelines:
            if verbose:
                tag = " [REBUILD]" if name in rebuild_set else ""
                print(f"\n=== VolGAN train/load for pipeline {name}{tag} ===")
            volgan_states[name] = train_volgan_for(
                name, surfaces[name], payload, paths=paths,
                quick=quick, rebuild=(name in rebuild_set), verbose=verbose,
            )
        phase2 = run_phase2(surfaces, volgan_states, payload, paths=paths, verbose=verbose)

    if phase1 and phase2:
        smd = write_summary_md(phase1, phase2, paths, phase1_price=phase1_price)
        if verbose:
            print(f"\nSummary written to {smd}")

    return {"phase1": phase1, "phase1_price": phase1_price, "phase2": phase2,
            "paths": paths, "model_states": model_states, "surfaces": surfaces}
