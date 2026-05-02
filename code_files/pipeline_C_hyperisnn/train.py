#!/usr/bin/env python
"""HyperISNN v3 — main entry point.

Trains one model per loss function (default: MSE and vega-weighted MSE),
evaluates on the held-out test set, runs constraint checks, measures inference
speed, and prints sample predictions. All output is mirrored to OUTPUT_FILE.

Run from the repo root::

    python -m HyperISNN.v3.train               # full run, default config
    python -m HyperISNN.v3.train --losses mse  # one loss only
    python -m HyperISNN.v3.train --steps 1000  # quick smoke test
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import load_or_build_cache, split_chronologically
from .evaluator import check_constraints, evaluate, sample_predictions, speed_test
from .losses import LOSS_REGISTRY, build_loss
from .model import HyperISNN
from .trainer import Trainer


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def make_logger(output_file: str):
    Path(output_file).write_text("")        # clear

    def log(msg: str = ""):
        print(msg, flush=True)
        with open(output_file, "a") as f:
            f.write(str(msg) + "\n")

    return log


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", nargs="+", default=["mse", "vega"],
                        choices=list(LOSS_REGISTRY.keys()),
                        help="Which losses to compare (default: mse vega)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total training steps (e.g. 1000 for quick test)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Override warm-up steps")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output log file path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the (slow) full test-set evaluation")
    args = parser.parse_args()

    cfg = Config()
    if args.steps is not None:
        cfg.train.n_steps = args.steps
    if args.warmup is not None:
        cfg.train.warmup_steps = args.warmup
    if args.batch is not None:
        cfg.train.batch_size = args.batch
    if args.output is not None:
        cfg.train.output_file = args.output
    if args.seed is not None:
        cfg.train.seed = args.seed

    log = make_logger(cfg.train.output_file)
    log(f"=== HyperISNN v3 started {datetime.now().isoformat(timespec='seconds')} ===")
    log(f"Data: {cfg.data.data_path} (price col: {cfg.data.price_col})")
    log(f"Losses to compare: {args.losses}")

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}   torch: {torch.__version__}")

    # ---- data ----
    prepared = load_or_build_cache(cfg.data, log_fn=log)
    train_data, val_data, test_data = split_chronologically(prepared, cfg.data)
    log(f"Split: train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    # ---- baseline (constant predictor) ----
    all_test_C = np.concatenate([d["C_norm"] for d in test_data])
    log(f"\n--- Sanity baselines (test set) ---")
    log(f"  Test C/S mean = {all_test_C.mean():.6f}  std = {all_test_C.std():.6f}")
    log(f"  Constant-predictor MSE on C/S = {all_test_C.var():.6f}  "
        f"(if val MSE ≈ this, model has collapsed to mean)")

    # ---- loop over losses ----
    results = {}
    for loss_name in args.losses:
        log(f"\n{'#'*72}")
        log(f"# Loss: {loss_name}")
        log(f"{'#'*72}")

        # Re-seed per loss so they all start from the SAME init
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

        model = HyperISNN(cfg)
        loss_fn = build_loss(loss_name, cfg.loss)

        trainer = Trainer(cfg, model, loss_fn, train_data, val_data, device, log_fn=log)
        trainer.train()

        if not args.no_eval:
            res = evaluate(model, test_data, cfg.hyper, device, name=f"test ({loss_name})", log_fn=log)
            results[loss_name] = res

            check_constraints(model, test_data, cfg.hyper, device, log_fn=log)
            speed_test(model, test_data, cfg.hyper, device, log_fn=log)
            sample_predictions(model, test_data, cfg.hyper, device, log_fn=log)

        # Save checkpoint
        ckpt_dir = Path(cfg.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"hyperisnn_v3_{loss_name}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config": cfg,
            "loss_name": loss_name,
            "history": trainer.history,
        }, ckpt_path)
        log(f"  saved checkpoint to {ckpt_path}")

    # ---- summary ----
    if results:
        log("\n========== SUMMARY ==========")
        log(f"{'loss':<10} {'price RMSE':>12} {'price MAE':>12} {'IV RMSE':>10} {'IV MAE':>10}")
        for k, v in results.items():
            log(f"{k:<10} {v['price_rmse']:12.4f} {v['price_mae']:12.4f} "
                f"{v['iv_rmse']:10.5f} {v['iv_mae']:10.5f}")
        log("=============================")

    log(f"\nDone @ {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
