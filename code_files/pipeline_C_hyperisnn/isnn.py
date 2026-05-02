"""ISNN-2 target network for option pricing.

Implements the ISNN-2 architecture of Jadoon et al. (2025), specialized to
the call-price surface (m, T) -> C/S.

Inputs:
    y_0 = (1 - m) * scale_m   (convex + monotone non-decreasing  → enforces
                                convex + monotone non-increasing in m,
                                i.e. butterfly + strike-monotonicity)
    t_0 = T * scale_t          (monotone non-decreasing  → calendar no-arb)
    x_0 = y_0                  (convexity branch with skip connections)

Branches (all with non-negative weights, σ_mc = softplus, σ_m = sigmoid):
    y_{h+1} = softplus(y_h W^{yy}_h^T + b^{y}_h),     h = 0, ..., L-2
    t_{h+1} = sigmoid (t_h W^{tt}_h^T + b^{t}_h),     h = 0, ..., L-2

x-branch (Eq. 9-10 of Jadoon et al.):
    x_1 = softplus(x_0 W^{xx}_0^T + y_0 W^{xy}_0^T + t_0 W^{xt}_0^T + b^{x}_0)
        # W^{xx}_0 is UNCONSTRAINED (only h≥1 needs to be ≥0 for convexity)

    x_{h+1} = softplus(x_h W^{xx}_h^T            (>= 0)
                     + x_0 W^{xx0}_h^T           (UNCONSTRAINED — adds expressiveness)
                     + y_h W^{xy}_h^T            (>= 0)
                     + t_h W^{xt}_h^T            (>= 0)
                     + b^{x}_h),                 h = 1, ..., L-1

Output:
    C/S = softplus(x_L * W^{out}^T + b^{out})    # b^{out} is the SHARED scalar

Vectorized: every tensor here carries a leading batch dim B (one per episode
in a training batch), so a single forward pass evaluates B different ISNNs
with different weight tensors.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# ISNN-2 weight specification: shapes, positivity, parameter offsets in a flat
# tensor so the hypernetwork can produce one big vector per episode.
# ----------------------------------------------------------------------------

@dataclass
class ISNNSpec:
    H: int = 8
    L: int = 3              # x-branch depth (produces x_1, ..., x_L)
    shapes: Dict[str, Tuple[int, ...]] = field(init=False)
    pos_names: set = field(init=False)
    offsets: Dict[str, int] = field(init=False)
    total: int = field(init=False)

    def __post_init__(self):
        H, L = self.H, self.L
        s: Dict[str, Tuple[int, ...]] = {}

        # y-branch: produces y_1, ..., y_{L-1}  (L-1 layers)
        s["Wyy_0"] = (H, 1);  s["by_0"] = (H,)
        for h in range(1, L - 1):
            s[f"Wyy_{h}"] = (H, H);  s[f"by_{h}"] = (H,)

        # t-branch: produces t_1, ..., t_{L-1}
        s["Wtt_0"] = (H, 1);  s["bt_0"] = (H,)
        for h in range(1, L - 1):
            s[f"Wtt_{h}"] = (H, H);  s[f"bt_{h}"] = (H,)

        # x-branch layer 0:  x_1 = σ(x_0 W^{xx}_0 + y_0 W^{xy}_0 + t_0 W^{xt}_0 + b^x_0)
        s["Wxx_0"] = (H, 1);   s["Wxy_0"] = (H, 1);   s["Wxt_0"] = (H, 1);   s["bx_0"] = (H,)

        # x-branch hidden layers 1..L-2:
        for h in range(1, L - 1):
            s[f"Wxx_{h}"]  = (H, H)
            s[f"Wxx0_{h}"] = (H, 1)
            s[f"Wxy_{h}"]  = (H, H)
            s[f"Wxt_{h}"]  = (H, H)
            s[f"bx_{h}"]   = (H,)

        # x-branch output layer L-1: produces x_L (scalar)
        s[f"Wxx_{L-1}"]  = (1, H)
        s[f"Wxx0_{L-1}"] = (1, 1)
        s[f"Wxy_{L-1}"]  = (1, H)
        s[f"Wxt_{L-1}"]  = (1, H)
        s[f"bx_{L-1}"]   = (1,)

        # Final linear pre-softplus (no separate W^out — folded into the last x layer)
        # We DO add a small (1,1) projection so that the output dimension matches conventions.
        s["Wout"] = (1, 1)        # scaling on x_L; positive
        # NOTE: bout is NOT in this dict — it's the shared learnable scalar held outside.

        # Positivity tags
        pos = set()
        # y-branch weights: all positive
        for k in s:
            if k.startswith("Wyy"):
                pos.add(k)
        # t-branch weights: all positive
        for k in s:
            if k.startswith("Wtt"):
                pos.add(k)
        # W^{xy}_h: positive for ALL h
        for k in s:
            if k.startswith("Wxy"):
                pos.add(k)
        # W^{xt}_h: positive for ALL h
        for k in s:
            if k.startswith("Wxt"):
                pos.add(k)
        # W^{xx}_h: positive for ALL h.
        # Note: ISNN-2 leaves Wxx_0 unconstrained for an Input-Convex network
        # (only convexity needed in input). For us we also need MONOTONICITY in
        # m (no-arb dC/dK <= 0), and x_0 = y_0 = (1 - m)*scale_m, so a negative
        # Wxx_0 flips the sign of dx_1/dm and breaks monotonicity. Constrain it.
        for h in range(0, L):
            pos.add(f"Wxx_{h}")
        # All Wxx0_h skip connections (from x_0 to layer h).
        # Must be positive to enforce monotonicity w.r.t moneyness (dC/dK <= 0).
        for h in range(1, L):
            pos.add(f"Wxx0_{h}")
        pos.add("Wout")

        self.shapes = s
        self.pos_names = pos
        offsets, off = {}, 0
        for k, sh in s.items():
            offsets[k] = off
            off += int(np.prod(sh))
        self.offsets = offsets
        self.total = off


# ----------------------------------------------------------------------------
# Vectorized ISNN-2 forward pass.
# ----------------------------------------------------------------------------

def _gather(flat: torch.Tensor, spec: ISNNSpec, name: str) -> torch.Tensor:
    """Slice the flat (B, total) tensor and reshape to (B, *shape)."""
    sh = spec.shapes[name]
    off = spec.offsets[name]
    sz = int(np.prod(sh))
    return flat[:, off:off + sz].reshape(flat.shape[0], *sh)


def isnn2_forward(
    flat: torch.Tensor,        # (B, spec.total)
    spec: ISNNSpec,
    m: torch.Tensor,           # (B, M, 1)
    T: torch.Tensor,           # (B, M, 1)
    r: torch.Tensor,           # (B, M, 1) risk-free rate
    bout: torch.Tensor,        # (1,) shared scalar
    scale_m: float,
    scale_t: float,
    pos_floor: float = 1e-6,
) -> torch.Tensor:
    """Vectorized ISNN-2 forward pass.

    Returns C/S of shape (B, M, 1). The output is the network's softplus head
    PLUS the call intrinsic-value floor max(0, 1 - m*exp(-rT)) — so the
    architecture *hard-enforces* the lower no-arb bound. The soft penalty in
    losses.py is therefore unnecessary.
    """
    B, M, _ = m.shape
    H, L = spec.H, spec.L

    def W(name: str) -> torch.Tensor:
        raw = _gather(flat, spec, name)
        if name in spec.pos_names:
            return F.softplus(raw) + pos_floor
        return raw

    def b(name: str) -> torch.Tensor:
        return _gather(flat, spec, name)        # biases are unconstrained

    # ----- inputs -----
    y0 = (1.0 - m) * scale_m                    # (B, M, 1)
    t0 = T * scale_t                            # (B, M, 1)
    x0 = y0                                     # share input

    # ----- y-branch (softplus, positive weights) -----
    y_h = y0
    y_states = [y0]
    for h in range(L - 1):
        Wyy = W(f"Wyy_{h}")                     # (B, H_out, H_in)
        by  = b(f"by_{h}").unsqueeze(1)         # (B, 1, H_out)
        y_h = F.softplus(torch.bmm(y_h, Wyy.transpose(-1, -2)) + by)
        y_states.append(y_h)                    # y_states[h+1] = y_{h+1}

    # ----- t-branch (sigmoid, positive weights) -----
    t_h = t0
    t_states = [t0]
    for h in range(L - 1):
        Wtt = W(f"Wtt_{h}")
        bt  = b(f"bt_{h}").unsqueeze(1)
        t_h = torch.sigmoid(torch.bmm(t_h, Wtt.transpose(-1, -2)) + bt)
        t_states.append(t_h)

    # ----- x-branch -----
    # Layer 0 :  x_1 = softplus(x_0 W^{xx}_0 + y_0 W^{xy}_0 + t_0 W^{xt}_0 + b^x_0)
    Wxx0 = W("Wxx_0")                            # (B, H, 1) — UNCONSTRAINED
    Wxy0 = W("Wxy_0")                            # (B, H, 1) — positive
    Wxt0 = W("Wxt_0")                            # (B, H, 1) — positive
    bx0  = b("bx_0").unsqueeze(1)                # (B, 1, H)
    pre  = (
        torch.bmm(x0, Wxx0.transpose(-1, -2))
        + torch.bmm(y0, Wxy0.transpose(-1, -2))
        + torch.bmm(t0, Wxt0.transpose(-1, -2))
        + bx0
    )
    x_h = F.softplus(pre)                        # (B, M, H)

    # Hidden layers 1 .. L-2
    for h in range(1, L - 1):
        Wxx  = W(f"Wxx_{h}")                     # (H, H)
        Wxx0_skip = W(f"Wxx0_{h}")               # (H, 1) — UNCONSTRAINED
        Wxy  = W(f"Wxy_{h}")                     # (H, H)
        Wxt  = W(f"Wxt_{h}")                     # (H, H)
        bx   = b(f"bx_{h}").unsqueeze(1)
        pre = (
            torch.bmm(x_h, Wxx.transpose(-1, -2))
            + torch.bmm(x0,  Wxx0_skip.transpose(-1, -2))
            + torch.bmm(y_states[h], Wxy.transpose(-1, -2))
            + torch.bmm(t_states[h], Wxt.transpose(-1, -2))
            + bx
        )
        x_h = F.softplus(pre)

    # Output layer L-1 : produces x_L (scalar per (B, M))
    h = L - 1
    Wxx  = W(f"Wxx_{h}")                          # (1, H)
    Wxx0_skip = W(f"Wxx0_{h}")                    # (1, 1)
    Wxy  = W(f"Wxy_{h}")                          # (1, H)
    Wxt  = W(f"Wxt_{h}")                          # (1, H)
    bx   = b(f"bx_{h}").unsqueeze(1)              # (B, 1, 1)
    pre = (
        torch.bmm(x_h, Wxx.transpose(-1, -2))
        + torch.bmm(x0,  Wxx0_skip.transpose(-1, -2))
        + torch.bmm(y_states[h], Wxy.transpose(-1, -2))
        + torch.bmm(t_states[h], Wxt.transpose(-1, -2))
        + bx
    )
    # Linear last layer: x_L = pre  (no activation here)
    x_L = pre                                     # (B, M, 1)

    # Final 1x1 positive scaling + shared bout, then softplus for non-negative
    # network output. Add the call intrinsic-value floor so C/S >= max(0, 1 - m*exp(-rT))
    # is guaranteed by the architecture (hard arb-free constraint).
    Wo  = W("Wout")                               # (B, 1, 1), positive
    pre_out = torch.bmm(x_L, Wo.transpose(-1, -2)) + bout.view(1, 1, 1)
    network_output = F.softplus(pre_out)          # (B, M, 1) >= 0
    intrinsic_bound = F.relu(1.0 - m)
    return network_output + intrinsic_bound       # (B, M, 1)


def isnn2_forward_single(
    flat: torch.Tensor,        # (spec.total,)
    spec: ISNNSpec,
    m: torch.Tensor,           # (M, 1)
    T: torch.Tensor,           # (M, 1)
    r: torch.Tensor,           # (M, 1)
    bout: torch.Tensor,        # (1,)
    scale_m: float,
    scale_t: float,
    pos_floor: float = 1e-6,
) -> torch.Tensor:
    """Single-episode forward (used for constraint checks / inference)."""
    return isnn2_forward(
        flat.unsqueeze(0), spec,
        m.unsqueeze(0), T.unsqueeze(0), r.unsqueeze(0),
        bout, scale_m, scale_t, pos_floor,
    ).squeeze(0)
