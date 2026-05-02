"""
=============================================================================
PIPELINE — HyperIV Training
=============================================================================
Trains the HyperIV model (SetEmbeddingNetwork + HyperNetwork) on SPX EOD
data prepared by pipeline_hyperiv_data.py.

Follows the training procedure from Yang et al. (ICML 2025):
  - MSE loss on implied volatility predictions
  - Arbitrage-free auxiliary loss (calendar spread + butterfly + integral)
  - Cosine annealing learning rate schedule
  - 500 epochs with batch size 128

Train/Val/Test split:
  - Train: 2013-01-02 to 2017-12-31
  - Val:   2018-01-01 to 2018-12-31
  - Test:  2019-01-01 to 2019-12-31
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import time
from tqdm import tqdm

from pipeline_B_hyperiv.hyperiv_util import create_hyperiv_model, HyperNetwork, build_iv_network
from pipeline_B_hyperiv.data_util import OptionDataset
from pipeline_B_hyperiv.trainer_util import trainer, aux_loss


# ============================================================
# 1. TRAINING LOOP WITH EVALUATION
# ============================================================
def train_hyperiv(
    train_df,
    val_df,
    # Architecture
    input_dim=3,
    hidden_dim=128,
    num_heads=2,
    num_layers=2,
    # Training
    num_epochs=500,
    batch_size=128,
    lr=1e-3,
    N_contracts=1024,
    # Auxiliary loss
    use_aux_loss=True,
    # Output
    save_dir=None,
    device=None,
):
    """
    Train HyperIV model on SPX EOD data.

    Args:
        train_df: DataFrame for training dates
        val_df: DataFrame for validation dates
        ... (architecture and training hyperparameters)

    Returns:
        model: trained HyperNetwork
        history: dict with loss curves
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("HYPERIV TRAINING")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Architecture: hidden_dim={hidden_dim}, heads={num_heads}, "
          f"layers={num_layers}")
    print(f"  Training: epochs={num_epochs}, batch_size={batch_size}, lr={lr}")
    print(f"  Aux loss: {use_aux_loss}")

    # Create datasets
    train_dataset = OptionDataset(train_df, N=N_contracts, sample=True)
    val_dataset = OptionDataset(val_df, N=N_contracts, sample=False)

    print(f"  Train dates: {len(train_dataset)}")
    print(f"  Val dates: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, drop_last=False
    )

    # Create model
    model, iv_network = create_hyperiv_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  IV network parameters: 337")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=1e-5
    )

    # Training history
    history = {
        "train_mse": [], "train_mae": [],
        "train_cal": [], "train_g": [], "train_integral": [],
        "val_mse": [], "val_mae": [],
        "val_cal": [], "val_g": [], "val_integral": [],
    }

    best_val_mae = float("inf")
    best_state = None

    start_time = time.time()

    for epoch in range(num_epochs):
        # --- Train ---
        train_results = trainer(
            train_loader, model, device, optimizer, is_train=True
        )
        train_mse, train_mae, train_cal, train_g, train_integral = train_results

        history["train_mse"].append(train_mse)
        history["train_mae"].append(train_mae)
        history["train_cal"].append(train_cal)
        history["train_g"].append(train_g)
        history["train_integral"].append(train_integral)

        lr_scheduler.step()

        # --- Validate ---
        val_results = trainer(
            val_loader, model, device, optimizer, is_train=False
        )
        val_mse, val_mae, val_cal, val_g, val_integral = val_results

        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_cal"].append(val_cal)
        history["val_g"].append(val_g)
        history["val_integral"].append(val_integral)

        # Track best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Logging
        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train MSE: {train_mse:.6f} MAE: {train_mae:.4f} | "
                f"Val MSE: {val_mse:.6f} MAE: {val_mae:.4f} | "
                f"Cal: {val_cal:.2e} G: {val_g:.2e} Int: {val_integral:.2e} | "
                f"LR: {current_lr:.6f} | {elapsed:.0f}s"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.0f}s")
    print(f"  Best val MAE: {best_val_mae:.6f}")

    # Save model
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.hyper_net.state_dict(), save_dir / "hyperiv_model.pth")
        torch.save(model.state_dict(), save_dir / "hyperiv_full_model.pth")

        with open(save_dir / "hyperiv_history.pkl", "wb") as f:
            pickle.dump(history, f)

        with open(save_dir / "hyperiv_config.pkl", "wb") as f:
            pickle.dump({
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "num_epochs": num_epochs,
                "best_val_mae": best_val_mae,
                "n_params": n_params,
            }, f)

        print(f"  Model saved to: {save_dir}")

    return model, history


# ============================================================
# 2. EVALUATION ON TEST SET
# ============================================================
def evaluate_hyperiv(model, test_df, device=None, N_contracts=None):
    """
    Evaluate trained HyperIV on test set.

    Returns:
        metrics: dict with MAE, MSE in IV and price space
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = OptionDataset(test_df, N=N_contracts or 2048, sample=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_results = trainer(test_loader, model, device, None, is_train=False)
    test_mse, test_mae, test_cal, test_g, test_integral = test_results

    metrics = {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_cal_loss": test_cal,
        "test_g_loss": test_g,
        "test_integral_loss": test_integral,
    }

    print(f"\n[HYPERIV TEST RESULTS]")
    print(f"  IV MAE: {test_mae:.6f}")
    print(f"  IV MSE: {test_mse:.6f}")
    print(f"  Calendar spread loss: {test_cal:.2e}")
    print(f"  Butterfly loss: {test_g:.2e}")
    print(f"  Integral loss: {test_integral:.2e}")

    return metrics


# ============================================================
# 3. WEIGHT VECTOR EXTRACTION
# ============================================================
def extract_weight_vectors(model, df, device=None, batch_size=64):
    """
    Extract HyperIV weight vectors omega_t for each date.

    For each date t, runs the hypernetwork on its 9 reference contracts
    to produce omega_t = g_theta(Z_t) in R^337.

    Args:
        model: trained HyperNetwork
        df: DataFrame with all dates (must have is_ref column)
        device: torch device
        batch_size: batch size for processing

    Returns:
        weight_vectors: np.ndarray shape (n_dates, 337)
        dates: list of dates (aligned with rows)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    dates = sorted(df["date"].unique())
    weight_vectors = []
    valid_dates = []

    print(f"\n  Extracting weight vectors for {len(dates)} dates...")

    for date_val in tqdm(dates, desc="  Extracting weights"):
        date_df = df[df["date"] == date_val]
        ref_df = date_df[date_df["is_ref"] == 1]

        if len(ref_df) < 3:
            continue

        # Build reference set: (k, t, sigma)
        z = ref_df[["log_moneyness", "tau", "implied_volatility"]].values
        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            omega = model.get_weights(z_tensor)  # (1, 337)

        weight_vectors.append(omega.cpu().numpy().flatten())
        valid_dates.append(date_val)

    weight_vectors = np.array(weight_vectors)
    print(f"  Extracted {len(weight_vectors)} weight vectors, shape: {weight_vectors.shape}")

    return weight_vectors, valid_dates


# ============================================================
# 4. SURFACE EVALUATION FROM WEIGHTS
# ============================================================
def evaluate_surface_from_weights(
    model, omega, k_grid, t_grid, device=None
):
    """
    Evaluate the IV surface on a grid using pre-computed weight vectors.

    Args:
        model: HyperNetwork (for functional forward)
        omega: (1, 337) or (337,) weight vector
        k_grid: 1D array of log-moneyness values
        t_grid: 1D array of time-to-maturity values (in years)
        device: torch device

    Returns:
        iv_surface: (len(t_grid), len(k_grid)) — IV values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    if omega.ndim == 1:
        omega = omega.reshape(1, -1)

    omega_tensor = torch.tensor(omega, dtype=torch.float32).to(device)

    # Create grid
    k_vals, t_vals = np.meshgrid(k_grid, t_grid)
    kt = np.stack([k_vals.ravel(), t_vals.ravel()], axis=1)
    kt_tensor = torch.tensor(kt, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        sigma = model.forward_from_weights(omega_tensor, kt_tensor)

    iv_surface = sigma.cpu().numpy().reshape(len(t_grid), len(k_grid))
    return iv_surface


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    output_dir = Path("./outputs/hyperiv")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prepared data
    print("Loading prepared HyperIV data...")
    df = pd.read_pickle(output_dir / "hyperiv_data.pkl")

    # Train/Val/Test split
    train_df = df[df["date"] < "2018-01-01"]
    val_df = df[(df["date"] >= "2018-01-01") & (df["date"] < "2019-01-01")]
    test_df = df[df["date"] >= "2019-01-01"]

    print(f"  Train: {train_df['date'].nunique()} dates, "
          f"Val: {val_df['date'].nunique()} dates, "
          f"Test: {test_df['date'].nunique()} dates")

    # Train
    model, history = train_hyperiv(
        train_df, val_df,
        num_epochs=500,
        batch_size=128,
        save_dir=output_dir,
    )

    # Evaluate on test
    metrics = evaluate_hyperiv(model, test_df)

    # Extract weight vectors for ALL dates
    weight_vectors, weight_dates = extract_weight_vectors(model, df)

    # Save
    with open(output_dir / "weight_vectors.pkl", "wb") as f:
        pickle.dump({
            "weight_vectors": weight_vectors,
            "dates": weight_dates,
        }, f)

    with open(output_dir / "test_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print("\n  Done!")
