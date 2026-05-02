"""
=============================================================================
CORRECTED PIPELINE — Part 3: ISNN-2 Model & Surface Fitting
=============================================================================
Implements the ISNN-2 architecture from Jadoon et al. (2025) — "Input
Specific Neural Networks" (arXiv:2503.00268v1).

Key corrections vs the original notebook:
  1. Normalization uses fwd_price (not Spot) for consistency with Moussa filter
  2. Softplus beta=1 (matching the paper's σ_mc = log(1 + exp(x)))
  3. MSE loss instead of aggressive Huber(beta=0.01)
  4. Proper weight projection after each optimizer step
  5. Output layer has NO activation (matching paper Remark 1 / Fig 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time


# ============================================================
# 0. NON-NEGATIVE LINEAR LAYER
# ============================================================
class NonNegativeLinear(nn.Linear):
    """
    Linear layer with weights clamped to be non-negative during forward pass.
    Required for preserving monotonicity/convexity in ISNN architecture.

    Paper Section 2 (ISNN-2):
    - W[xx]_h must be non-negative for h >= 1  (convexity in x0)
    - W[yy]_h, W[xy]_h must be non-negative    (convex + monotone in y0)
    - W[tt]_h, W[xt]_h must be non-negative    (monotone in t0)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self._init_positive()

    def _init_positive(self):
        """Initialize with strictly positive weights."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.abs_()  # Force positive
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.clamp(min=0), self.bias)


# ============================================================
# 1. ISNN-2 ARCHITECTURE
# ============================================================
class ISNN2_OptionSurface(nn.Module):
    """
    ISNN-2 for option price surfaces.

    Input mapping for options:
      x0 = normalized strike (K/F)  → CONVEX  (call prices are convex in strike)
      t0 = normalized time (T/365)  → MONOTONE NON-DECREASING (prices increase with T)

    Architecture (Paper Eqs 6-10, Figure 2):
      - Monotonic branch: t_h+1 = σ_m(t_h W[tt]^T + b[t])
      - Convex branch:    x_1   = σ_mc(x_0 W[xx0]^T + t_0 W[xt0]^T + b[x0])
                          x_h+1 = σ_mc(x_h W[xx]^T + x_0 W[xx0]^T + t_h W[xt]^T + b[x])
      - Output:           x_H   = F_{H-1}  (NO activation on final layer per Remark 1)

    Constraints:
      - W[xx]_h >= 0 for h >= 1     (propagates convexity)
      - W[tt]_h >= 0                (monotonicity in time)
      - W[xt]_h >= 0                (non-negative injection of monotone state)
      - W[xx0]_h can be ANY sign    (skip connections preserve convexity regardless)
      - σ_mc = Softplus (convex & monotonically non-decreasing)
      - σ_m  = Softplus (monotonically non-decreasing; also convex, which is fine)
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.H = num_layers  # Total layers (including input processing)

        # === MONOTONIC BRANCH (Time t0) ===
        # Eq (8): t_{h+1} = σ_m(t_h W[tt]^T + b[t])
        # All W[tt] must be non-negative
        self.t_layers = nn.ModuleList()
        self.t_layers.append(NonNegativeLinear(1, hidden_dim))
        for _ in range(num_layers - 2):
            self.t_layers.append(NonNegativeLinear(hidden_dim, hidden_dim))

        # === CONVEX BRANCH (Strike x0) ===
        # Eq (9): x_1 = σ_mc(x_0 W[xx0]^T + t_0 W[xt0]^T + b[x])
        self.w0_x = nn.Linear(1, hidden_dim)           # W[xx]_0: can be any sign
        self.w0_t = NonNegativeLinear(1, hidden_dim)    # W[xt]_0: must be >= 0

        # Eq (10): x_{h+1} = σ_mc(x_h W[xx]^T + x_0 W[xx0]^T + t_h W[xt]^T + b[x])
        self.x_layers_xx = nn.ModuleList()    # W[xx]: non-negative (propagate convexity)
        self.x_layers_xx0 = nn.ModuleList()   # W[xx0]: any sign (skip connections)
        self.x_layers_xt = nn.ModuleList()    # W[xt]: non-negative (inject monotone state)

        for _ in range(num_layers - 2):
            self.x_layers_xx.append(NonNegativeLinear(hidden_dim, hidden_dim))
            self.x_layers_xx0.append(nn.Linear(1, hidden_dim))       # Skip: any sign
            self.x_layers_xt.append(NonNegativeLinear(hidden_dim, hidden_dim))

        # === OUTPUT LAYER ===
        # Must be non-negative to aggregate convex states
        # Paper Remark 1: NO activation on the last layer
        self.w_out = NonNegativeLinear(hidden_dim, 1, bias=False)

    def forward(self, x0, t0):
        """
        Forward pass.

        Args:
            x0: (batch, 1) normalized strike (K/F)
            t0: (batch, 1) normalized time (T/365)

        Returns:
            (batch, 1) normalized price
        """
        # Activation: Softplus with beta=1 (paper Eq 11: σ_mc(x) = log(1 + exp(x)))
        act = nn.Softplus(beta=1)

        # === 1. Monotonic Branch ===
        t_states = []
        t_h = t0
        for layer in self.t_layers:
            t_h = act(layer(t_h))
            t_states.append(t_h)

        # === 2. Convex Branch ===
        # Layer 0: Eq (9)
        z_h = act(self.w0_x(x0) + self.w0_t(t0))

        # Layers 1..H-2: Eq (10)
        for i in range(self.H - 2):
            term_xx = self.x_layers_xx[i](z_h)       # Propagate convexity
            term_xx0 = self.x_layers_xx0[i](x0)      # Skip from input strike
            term_xt = self.x_layers_xt[i](t_states[i])  # Inject monotone time state
            z_h = act(term_xx + term_xx0 + term_xt)

        # === 3. Output (NO activation — Remark 1) ===
        price = self.w_out(z_h)

        return price


# ============================================================
# 2. DATA PREPARATION
# ============================================================
class OptionSurfaceData:
    """
    Prepares a single date's option data for ISNN training.

    CRITICAL NORMALIZATION (corrected):
    - K_norm = strike / fwd_price  (moneyness, consistent with Moussa)
    - T_norm = maturity_days / 365
    - C_norm = moussa_price / (discount * fwd_price)  (consistent with Moussa's C̃)

    This ensures the ISNN trains in the SAME normalized space
    that the Moussa filter operates in.
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()

        # Use per-row fwd_price and discount for normalization
        # (they vary by expiry within the same date)
        self.fwd = self.df["fwd_price"].values
        self.disc = self.df["discount"].values
        norm_factor = self.fwd * self.disc  # B(T)*F(T) per row

        # Moneyness (strike / forward)
        k_norm = self.df["strike"].values / self.fwd

        # Normalized time
        t_norm = self.df["maturity_days"].values / 365.0

        # Normalized price (same as Moussa's C_norm)
        c_norm = self.df["moussa_price"].values / norm_factor

        # Store for denormalization
        self.norm_factor = norm_factor

        # Convert to tensors
        self.x_strike = torch.tensor(k_norm, dtype=torch.float32).unsqueeze(1)
        self.t_time = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1)
        self.y_price = torch.tensor(c_norm, dtype=torch.float32).unsqueeze(1)

    def get_tensors(self):
        return self.x_strike, self.t_time, self.y_price

    def denormalize_price(self, c_norm_tensor):
        """Convert normalized model output back to dollar prices."""
        nf = torch.tensor(self.norm_factor, dtype=torch.float32).unsqueeze(1)
        return c_norm_tensor * nf


# ============================================================
# 3. TRAINING
# ============================================================
def project_weights(model):
    """
    Project all NonNegativeLinear weights to be strictly positive.
    Called after each optimizer step to maintain architecture constraints.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, NonNegativeLinear):
                module.weight.data.clamp_(min=1e-6)


def fit_surface_for_date(
    date_key,
    dataframe,
    epochs=2000,
    lr=0.005,
    hidden_dim=128,
    num_layers=3,
    verbose=True,
):
    """
    Fit the ISNN-2 model for a single date's option surface.

    Args:
        date_key: date identifier (for logging)
        dataframe: DataFrame from the Moussa-filtered surface dict
        epochs: training iterations
        lr: learning rate for Adam
        hidden_dim: neurons per hidden layer
        num_layers: total depth
        verbose: print training progress

    Returns:
        model: trained ISNN2_OptionSurface
        processor: OptionSurfaceData instance
        loss_history: list of loss values
    """
    # --- Prepare Data ---
    processor = OptionSurfaceData(dataframe)
    x_train, t_train, y_train = processor.get_tensors()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    t_train = t_train.to(device)
    y_train = y_train.to(device)

    # --- Initialize Model ---
    model = ISNN2_OptionSurface(hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Use MSE loss (smoother gradients than aggressive Huber)
    criterion = nn.MSELoss()
    # Learning rate schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=200, factor=0.5, min_lr=1e-6
    )

    # --- Training Loop ---
    if verbose:
        print(f"[{date_key}] Training on {len(x_train)} contracts | dim={hidden_dim}, layers={num_layers}")

    start_time = time.time()
    loss_history = []
    best_loss = float("inf")
    best_state = None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = model(x_train, t_train)
        loss = criterion(predictions, y_train)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Project weights to maintain constraints
        project_weights(model)

        # Track best model
        loss_val = loss.item()
        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Update scheduler (use .detach() to avoid the requires_grad warning)
        scheduler.step(loss.detach())

        if verbose and (epoch + 1) % 500 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss_val:.8f} | LR: {current_lr:.6f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - start_time
    if verbose:
        print(f"[{date_key}] Done in {train_time:.2f}s | Best Loss: {best_loss:.8f}")

    return model, processor, loss_history


# ============================================================
# 4. VALIDATION: Check arbitrage-free constraints on the model
# ============================================================
def validate_model_constraints(model, k_range=(0.5, 2.0), t_range=(0.01, 3.0), grid_size=50):
    """
    Numerically verify that the trained ISNN satisfies:
    1. Monotonicity: dC/dT >= 0  (price increases with time)
    2. Convexity:    d²C/dK² >= 0 (price is convex in strike)

    Uses autograd to compute derivatives on a dense grid.
    """
    model.eval()

    grid_k = torch.linspace(k_range[0], k_range[1], grid_size)
    grid_t = torch.linspace(t_range[0], t_range[1], grid_size)
    G_k, G_t = torch.meshgrid(grid_k, grid_t, indexing="ij")

    x_flat = G_k.flatten().unsqueeze(1).requires_grad_(True)
    t_flat = G_t.flatten().unsqueeze(1).requires_grad_(True)

    price = model(x_flat, t_flat)
    grads = torch.autograd.grad(price, [x_flat, t_flat], torch.ones_like(price), create_graph=True)
    dC_dK = grads[0]
    dC_dT = grads[1]
    d2C_dK2 = torch.autograd.grad(dC_dK, x_flat, torch.ones_like(dC_dK), create_graph=False)[0]

    eps = -1e-6  # tolerance
    time_violations = (dC_dT < eps).sum().item()
    convex_violations = (d2C_dK2 < eps).sum().item()
    total_points = x_flat.shape[0]

    results = {
        "total_points": total_points,
        "time_monotonicity_violations": time_violations,
        "strike_convexity_violations": convex_violations,
        "is_valid": time_violations == 0 and convex_violations == 0,
    }

    return results


# ============================================================
# 5. BATCH FITTING (all dates)
# ============================================================
def fit_all_surfaces(
    filtered_dict,
    epochs=2000,
    lr=0.005,
    hidden_dim=128,
    num_layers=3,
    max_dates=None,
    validate=True,
):
    """
    Fit ISNN models for all dates in the filtered dictionary.

    Args:
        filtered_dict: output of moussa_filter_surface
        epochs, lr, hidden_dim, num_layers: training hyperparameters
        max_dates: limit number of dates (for testing)
        validate: run constraint validation on each model

    Returns:
        models_dict: dict {date -> (model, processor, losses)}
        validation_results: dict {date -> validation_dict}
    """
    print("\n" + "=" * 60)
    print("ISNN-2 SURFACE FITTING")
    print("=" * 60)

    dates = sorted(filtered_dict.keys())
    if max_dates is not None:
        dates = dates[:max_dates]

    models_dict = {}
    validation_results = {}
    fit_times = []

    for i, d in enumerate(dates):
        df = filtered_dict[d]

        if len(df) < 5:
            print(f"  [{d.date()}] Skipping — only {len(df)} quotes")
            continue

        model, processor, losses = fit_surface_for_date(
            date_key=d.date(),
            dataframe=df,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            verbose=(i < 3 or (i + 1) % 100 == 0),  # Verbose for first few + periodic
        )
        models_dict[d] = (model, processor, losses)

        if validate:
            # Determine actual data range for validation grid
            k_min = df["K_norm"].min() if "K_norm" in df.columns else 0.5
            k_max = df["K_norm"].max() if "K_norm" in df.columns else 2.0
            t_min = df["maturity_days"].min() / 365.0
            t_max = df["maturity_days"].max() / 365.0

            val = validate_model_constraints(
                model,
                k_range=(max(0.3, k_min * 0.9), min(3.0, k_max * 1.1)),
                t_range=(max(0.001, t_min * 0.9), min(5.0, t_max * 1.1)),
            )
            validation_results[d] = val

            if not val["is_valid"] and i < 10:
                print(f"    ⚠️  Violations: {val['time_monotonicity_violations']} time, "
                      f"{val['strike_convexity_violations']} convexity")

    # Summary
    if validate and validation_results:
        n_valid = sum(1 for v in validation_results.values() if v["is_valid"])
        n_total = len(validation_results)
        print(f"\n[VALIDATION SUMMARY]")
        print(f"  Arbitrage-free surfaces: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    return models_dict, validation_results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    from pipeline_part1_preprocessing import load_all_data, preprocess
    from pipeline_part2_moussa_filter import moussa_filter_surface, run_diagnostics

    # Load & preprocess
    data = load_all_data(".")
    surface_dict = preprocess(data, min_volume=10)

    # Filter
    filtered_dict, stats = moussa_filter_surface(surface_dict, lambda_param=0.01)
    run_diagnostics(filtered_dict)

    # Fit a single date as test
    sample_date = sorted(filtered_dict.keys())[5]
    sample_df = filtered_dict[sample_date]

    model, processor, losses = fit_surface_for_date(
        date_key=sample_date.date(),
        dataframe=sample_df,
        epochs=2000,
        lr=0.005,
        hidden_dim=128,
        num_layers=3,
    )

    # Validate
    val = validate_model_constraints(model)
    print(f"\n[Constraint Validation]")
    print(f"  Grid points: {val['total_points']}")
    print(f"  Time monotonicity violations: {val['time_monotonicity_violations']}")
    print(f"  Strike convexity violations:  {val['strike_convexity_violations']}")
    print(f"  {'✅ PASSED' if val['is_valid'] else '⚠️  VIOLATIONS DETECTED'}")

    # Plot loss
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.yscale("log")
        plt.title(f"Training Loss — {sample_date.date()}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (log)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_loss.png", dpi=150)
        plt.show()
        print("Saved training_loss.png")
    except ImportError:
        print("matplotlib not available — skipping plot")