"""Compare HyperISNN against simple non-ML baselines on the test set.

Baselines (all use the SAME (reference, target) episode split per day so
comparison is fair):
  B0: constant predictor — predict the train-mean C/S for everything
  B1: per-day mean — predict the day's reference-set mean C/S
  B2: kNN in (m, T) — nearest-neighbour from the day's reference set
  B3: per-day linear regression on (m, T) → C/S
  B4: per-day BS implied vol from references, BS price for targets

Metric: MSE on C/S, RMSE on price ($).
"""

import numpy as np
import torch

from .bs_utils import bs_call, implied_vol
from .config import Config
from .data import load_or_build_cache, split_chronologically
from .model import HyperISNN
from .trainer import sample_episode


def evaluate_baseline(name, predict_fn, dataset, cfg_hyper, train_mean=None):
    """predict_fn(day, c_idx, t_idx, train_mean) → predictions for target set."""
    rng = np.random.default_rng(0)
    sq, st = [], []
    for d in dataset:
        c_idx, t_idx = sample_episode(d, cfg_hyper, rng)
        pred = predict_fn(d, c_idx, t_idx, train_mean)
        true = d["C_norm"][t_idx]
        sq.append(((pred - true) ** 2).mean())
        st.append(true)
    mse = float(np.mean(sq))
    print(f"  {name:30s}: MSE = {mse:.6f}  RMSE(C/S) = {np.sqrt(mse):.5f}")
    return mse


def b_constant(d, c_idx, t_idx, train_mean):
    return np.full(len(t_idx), train_mean, dtype=np.float32)


def b_day_mean(d, c_idx, t_idx, _):
    return np.full(len(t_idx), d["C_norm"][c_idx].mean(), dtype=np.float32)


def b_knn(d, c_idx, t_idx, _):
    # 3-NN in (logm, T) feature space
    F_ref = np.stack([d["logm"][c_idx], d["T"][c_idx] * 5.0], axis=1)
    F_tgt = np.stack([d["logm"][t_idx], d["T"][t_idx] * 5.0], axis=1)
    C_ref = d["C_norm"][c_idx]
    preds = np.zeros(len(t_idx), dtype=np.float32)
    K = min(3, len(c_idx))
    for i in range(len(t_idx)):
        dist = np.sum((F_ref - F_tgt[i]) ** 2, axis=1)
        idx = np.argsort(dist)[:K]
        preds[i] = C_ref[idx].mean()
    return preds


def b_linreg(d, c_idx, t_idx, _):
    # Fit C_norm = a + b*logm + c*T  via least squares
    X = np.column_stack([np.ones(len(c_idx)), d["logm"][c_idx], d["T"][c_idx]])
    y = d["C_norm"][c_idx]
    if X.shape[0] < 4:
        return np.full(len(t_idx), y.mean(), dtype=np.float32)
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt = np.column_stack([np.ones(len(t_idx)), d["logm"][t_idx], d["T"][t_idx]])
    return np.maximum(0, Xt @ coeff).astype(np.float32)


def b_bs_flatvol(d, c_idx, t_idx, _):
    # Take median IV from references, price targets at that vol with BS
    iv_ref = d["iv"][c_idx]
    iv_ref = iv_ref[~np.isnan(iv_ref)]
    if len(iv_ref) == 0:
        return np.full(len(t_idx), 0.013, dtype=np.float32)
    sigma = float(np.median(iv_ref))
    K = d["K"][t_idx]; T = d["T"][t_idx]
    C_dollar = bs_call(d["S"], K, T, d["r"], sigma)
    return (C_dollar / d["S"]).astype(np.float32)


def b_hyperisnn(model, cfg_hyper, device):
    """Returns a predict_fn closure for the trained HyperISNN model."""
    from .trainer import collate, to_device
    def predict(d, c_idx, t_idx, _):
        b = to_device(collate([(d, c_idx, t_idx)], cfg_hyper), device)
        with torch.no_grad():
            pred = model(b["q"], b["mask"], b["ctx"], b["m"], b["T"], b["r"]).cpu().numpy().squeeze()
        if pred.ndim == 0: pred = pred.reshape(1)
        return pred[:len(t_idx)]
    return predict


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = load_or_build_cache(cfg.data)
    train_data, val_data, test_data = split_chronologically(prepared, cfg.data)
    train_mean = float(np.concatenate([d["C_norm"] for d in train_data]).mean())

    print(f"\n=== Baselines on TEST set ({len(test_data)} days) ===")
    print(f"  train mean C/S = {train_mean:.5f}")

    evaluate_baseline("B0: global constant",         b_constant,    test_data, cfg.hyper, train_mean)
    evaluate_baseline("B1: per-day reference mean",  b_day_mean,    test_data, cfg.hyper)
    evaluate_baseline("B2: per-day 3-NN in (m,T)",   b_knn,         test_data, cfg.hyper)
    evaluate_baseline("B3: per-day linreg(m,T)",     b_linreg,      test_data, cfg.hyper)
    evaluate_baseline("B4: per-day BS flat-vol",     b_bs_flatvol,  test_data, cfg.hyper)

    # HyperISNN
    print()
    model = HyperISNN(cfg).to(device)
    ckpt = torch.load("checkpoints_v3/hyperisnn_v3_mse.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    evaluate_baseline("M : HyperISNN v3",            b_hyperisnn(model, cfg.hyper, device), test_data, cfg.hyper)


if __name__ == "__main__":
    main()
