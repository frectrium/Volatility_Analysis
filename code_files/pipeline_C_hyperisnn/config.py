"""Centralized configuration for HyperISNN v3.

Every hyperparameter lives here so experiments are reproducible and
ablations are a one-line change.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class DataConfig:
    """Where to find the data and how to filter it for training."""
    data_path: Path = Path("data/shared/filtered_dict.pkl")
    market_path: Path = Path("data/shared/market_data.pkl")
    cache_path: Path = Path("data/pipeline_d/hyperisnn_prepared_cache.pkl")  # reuse v2 cache
    price_col: str = "moussa_price"

    # Per-day filtering
    moneyness_lo: float = 0.7
    moneyness_hi: float = 1.3
    maturity_lo_days: float = 7.0
    maturity_hi_days: float = 365.0
    min_quotes_per_day: int = 30

    # Forward-curve grid (from market_data['fwd_df'])
    tenors_days: Tuple[float, ...] = (0, 7, 14, 30, 60, 90, 180, 350)

    # Train / val / test split (chronological)
    train_frac: float = 0.70
    val_frac: float = 0.10


@dataclass
class ISNNConfig:
    """ISNN-2 target network architecture."""
    H: int = 8                       # hidden width per layer
    L: int = 3                       # x-branch depth (produces x_1 .. x_L)
    # Sweep history (suggestions.md Phase 3 #1):
    #   H=32 was tested 2026-04-29 — strictly worse than H=8 on both IV and price RMSE
    #   (warmup over-saturates and the wider ISNN never recovers). Reverted.
    # Branches have L-1 internal layers (produce y_1, ..., y_{L-1})

    # Input scaling (keeps activations in the curved region)
    scale_m: float = 3.0             # y_0 = (1-m) * scale_m
    scale_t: float = 2.0             # t_0 = T   * scale_t
    # x_0 uses the same scaling as y_0


@dataclass
class HyperConfig:
    """HyperNetwork architecture."""
    d_hyper: int = 128
    n_heads: int = 4
    n_layers: int = 3                # transformer encoder layers
    head_hidden: int = 256           # MLP head intermediate dim
    head_activation: str = "gelu"
    transformer_dropout: float = 0.0

    # Reference set
    n_ctx_min: int = 8
    n_ctx_max: int = 18
    n_tgt_max: int = 80

    # Init: std of the final Linear that produces ω (10x larger than pre-fix
    # so per-day modulation is non-trivial from step 1)
    head_init_std: float = 0.1


@dataclass
class ResidualConfig:
    """Residual weight-generation scheme:  W_raw = W_base + alpha * omega."""
    alpha_init: float = 0.1          # per-parameter modulation scale at init (10x larger so hypernet matters)
    pos_floor: float = 1e-6          # softplus(raw) + floor for positive weights

    # W_base targets (the value softplus(raw) should equal at init)
    target_input_W: float = 1.0      # W^{yy}_0, W^{tt}_0, W^{xy}_0, W^{xt}_0
    target_hidden_W: float = 0.0     # 0 → auto-resolve to 1/H via Config.resolve_target_hidden_W (so widening H rescales correctly)
    target_output_W: float = 0.5     # W^{xx}_L, W^{xy}_L, W^{xt}_L (last layer); larger so x_L spans wider range

    # Unconstrained W_base init (Gaussian std)
    free_input_std: float = 1.0      # W^{xx}_0  (now constrained — see isnn.ISNNSpec)
    free_skip_std: float = 0.5       # W^{xx0}_h (now constrained — see isnn.ISNNSpec)
    free_bias_init: float = 0.0      # all biases

    # Final shift: bout_init = -10 gives softplus(-10) = 4.5e-5 floor so model
    # can predict near-zero OTM prices. Combined with larger target_output_W
    # so Wo*x_L can reach ~ +5..+8 to give ATM/ITM prices.
    bout_init: float = -10.0


@dataclass
class LossConfig:
    """Loss-function weights.

    Note: the intrinsic-value penalty is gone — the architecture (isnn.py)
    now hard-enforces C/S >= max(0, 1 - m*exp(-rT)).
    """
    lambda_upper: float = 1.0        # max(0, C/S - 1)
    lambda_var: float = 0.01         # anti-collapse safety net


@dataclass
class TrainConfig:
    """Training loop hyperparameters."""
    seed: int = 42
    batch_size: int = 32
    n_steps: int = 10000
    warmup_steps: int = 500          # train only W_base + bout (alpha frozen)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Scheduler
    scheduler_T0: int = 1000
    scheduler_Tmult: int = 2
    scheduler_eta_min: float = 1e-5

    # Logging
    log_every: int = 250
    val_every: int = 250
    output_file: str = "output_v3.txt"
    checkpoint_dir: Path = Path("data/pipeline_d/checkpoints_v3")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    isnn: ISNNConfig = field(default_factory=ISNNConfig)
    hyper: HyperConfig = field(default_factory=HyperConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def resolve_target_hidden_W(self) -> float:
        """Default hidden W target = 1/H if not overridden."""
        if self.residual.target_hidden_W <= 0:
            return 1.0 / self.isnn.H
        return self.residual.target_hidden_W


def softplus_inverse(y: float) -> float:
    """Returns x such that softplus(x) = y, i.e. x = log(exp(y) - 1)."""
    if y <= 0:
        raise ValueError(f"softplus_inverse requires y > 0, got {y}")
    if y > 20:
        return y                     # numerically stable approximation
    return math.log(math.expm1(y))
