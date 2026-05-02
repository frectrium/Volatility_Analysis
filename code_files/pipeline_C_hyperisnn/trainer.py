"""Training loop with warm-up and rich diagnostics.

Episodic training: each step samples a batch of (day, context, target) triples,
runs the hypernetwork forward, evaluates the ISNN on the target points, and
takes one optimizer step.

Warm-up phase (first cfg.train.warmup_steps): the hypernetwork is FROZEN and
α is held at its init value. Only W_base and b^{out} train. This forces the
model to find a good "average" surface before allowing per-day modulation.
"""

import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, HyperConfig
from .losses import BaseLoss, _per_episode_variance
from .model import HyperISNN


# ----------------------------------------------------------------------------
# Episode sampling and batching
# ----------------------------------------------------------------------------

def sample_episode(day: dict, cfg: HyperConfig, rng: np.random.Generator):
    """Pick a random (context, target) split from the day's quotes."""
    n_q = len(day["K"])
    n_ctx = int(rng.integers(cfg.n_ctx_min, min(cfg.n_ctx_max, n_q) + 1))
    n_ctx = min(n_ctx, n_q)
    perm = rng.permutation(n_q)
    c_idx = perm[:n_ctx]
    rest = perm[n_ctx:]
    n_t = min(cfg.n_tgt_max, len(rest))
    t_idx = rest[:n_t]
    return c_idx, t_idx


def collate(eps: List[Tuple[dict, np.ndarray, np.ndarray]], cfg: HyperConfig) -> Dict[str, torch.Tensor]:
    """Pad-and-stack a list of (day, c_idx, t_idx) into batched tensors."""
    B = len(eps)
    M = max(len(t_idx) for _, _, t_idx in eps)

    q_pad = np.zeros((B, cfg.n_ctx_max, 3), dtype=np.float32)
    cmask = np.zeros((B, cfg.n_ctx_max),   dtype=bool)
    ctx   = np.zeros((B, 9),               dtype=np.float32)
    tgt_m   = np.zeros((B, M, 1),          dtype=np.float32)
    tgt_T   = np.zeros((B, M, 1),          dtype=np.float32)
    tgt_C   = np.zeros((B, M, 1),          dtype=np.float32)
    tgt_v   = np.zeros((B, M, 1),          dtype=np.float32)
    tgt_msk = np.zeros((B, M, 1),          dtype=np.float32)
    tgt_S   = np.zeros((B, 1, 1),          dtype=np.float32)
    tgt_r   = np.zeros((B, M, 1),          dtype=np.float32)

    for i, (day, c_idx, t_idx) in enumerate(eps):
        n = len(c_idx)
        q_pad[i, :n, 0] = day["logm"][c_idx]
        q_pad[i, :n, 1] = day["T"][c_idx]
        q_pad[i, :n, 2] = day["C_norm"][c_idx]
        cmask[i, :n] = True

        # Centred forward curve + rate
        ctx[i, :8] = (day["fwd_curve"] / day["S"] - 1.0) * 10.0
        ctx[i,  8] = day["r"] * 10.0

        nt = len(t_idx)
        tgt_m[i, :nt, 0] = day["m"][t_idx]
        tgt_T[i, :nt, 0] = day["T"][t_idx]
        tgt_C[i, :nt, 0] = day["C_norm"][t_idx]
        tgt_v[i, :nt, 0] = day["vega"][t_idx] / day["S"]
        tgt_msk[i, :nt, 0] = 1.0
        tgt_S[i, 0, 0] = day["S"]
        tgt_r[i, :nt, 0] = day["r"]

    return dict(
        q=torch.from_numpy(q_pad),
        mask=torch.from_numpy(cmask),
        ctx=torch.from_numpy(ctx),
        m=torch.from_numpy(tgt_m),
        T=torch.from_numpy(tgt_T),
        C=torch.from_numpy(tgt_C),
        vega=torch.from_numpy(tgt_v),
        tmask=torch.from_numpy(tgt_msk),
        S=torch.from_numpy(tgt_S),
        r=torch.from_numpy(tgt_r),
    )


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: HyperISNN,
        loss_fn: BaseLoss,
        train_data: List[dict],
        val_data: List[dict],
        device: torch.device,
        log_fn: Callable[[str], None] = print,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.log = log_fn

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt,
            T_0=cfg.train.scheduler_T0,
            T_mult=cfg.train.scheduler_Tmult,
            eta_min=cfg.train.scheduler_eta_min,
        )

        self.rng = np.random.default_rng(cfg.train.seed)
        self.history: List[dict] = []

    def _sample_batch(self, dataset: List[dict], batch_size: int) -> Dict[str, torch.Tensor]:
        idx = self.rng.integers(0, len(dataset), size=batch_size)
        eps = [(dataset[i], *sample_episode(dataset[i], self.cfg.hyper, self.rng)) for i in idx]
        return to_device(collate(eps, self.cfg.hyper), self.device)

    @torch.no_grad()
    def _dead_positive_pct(self, batch: Dict[str, torch.Tensor]) -> float:
        """Fraction of positive ISNN weights that hit the pos_floor.

        We compute the actual softplus(W_raw)+floor for one batch and check
        how many entries are within 1% of the floor (i.e. effectively dead).
        """
        flat = self.model.generate_flat_weights(
            batch["q"], batch["mask"], batch["ctx"]
        )
        spec = self.model.spec
        floor = self.model.cfg.residual.pos_floor
        thresh = floor * 1.01
        total, dead = 0, 0
        import torch.nn.functional as _F
        for name in spec.pos_names:
            sh = spec.shapes[name]
            off = spec.offsets[name]
            sz = int(np.prod(sh))
            raw = flat[:, off:off + sz]
            w = _F.softplus(raw) + floor
            total += w.numel()
            dead += (w <= thresh).sum().item()
        return 100.0 * dead / max(total, 1)

    def _eval_val(self, n_chunks: int = 6) -> Dict[str, float]:
        """Quick validation pass on a few chunks (fast but representative)."""
        self.model.eval()
        with torch.no_grad():
            total_sq, total_n = 0.0, 0.0
            pred_var_sum, true_var_sum, var_n = 0.0, 0.0, 0
            chunk_size = 32
            for j in range(0, min(len(self.val_data), n_chunks * chunk_size), chunk_size):
                chunk = self.val_data[j:j + chunk_size]
                eps = [(d, *sample_episode(d, self.cfg.hyper, self.rng)) for d in chunk]
                b = to_device(collate(eps, self.cfg.hyper), self.device)
                pred = self.model(b["q"], b["mask"], b["ctx"], b["m"], b["T"], b["r"])
                sq = (((pred - b["C"]) ** 2) * b["tmask"]).sum().item()
                total_sq += sq
                total_n += b["tmask"].sum().item()
                pv = _per_episode_variance(pred, b["tmask"])
                tv = _per_episode_variance(b["C"], b["tmask"])
                pred_var_sum += pv.sum().item()
                true_var_sum += tv.sum().item()
                var_n += pv.numel()
        self.model.train()
        return {
            "val_mse": total_sq / max(total_n, 1.0),
            "pred_var": pred_var_sum / max(var_n, 1),
            "true_var": true_var_sum / max(var_n, 1),
        }

    def train(self) -> Dict[str, list]:
        cfg = self.cfg.train
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.log(f"\n{'='*64}")
        self.log(f"Training with loss '{self.loss_fn.name}'")
        self.log(f"{'='*64}")
        self.log(f"  total params : {sum(p.numel() for p in self.model.parameters()):,}")
        self.log(f"  ISNN params  : {self.model.spec.total} (per-day)")
        self.log(f"  warm-up      : {cfg.warmup_steps} steps (alpha & hypernet frozen)")
        self.log(f"  total steps  : {cfg.n_steps}")
        self.log(f"  optimizer    : AdamW(lr={cfg.lr}, wd={cfg.weight_decay})")
        self.log(f"  batch size   : {cfg.batch_size}")

        # Start in warm-up mode
        self.model.freeze_alpha()
        warmup_active = True
        self.log(f"  >>> WARM-UP PHASE — only W_base & b^out train")

        loss_window = deque(maxlen=cfg.log_every)
        comps_window: Dict[str, deque] = {}
        ratio_window = deque(maxlen=cfg.log_every)

        t0 = time.time()
        self.model.train()

        for step in range(cfg.n_steps):
            # Switch to full training after warm-up
            if warmup_active and step >= cfg.warmup_steps:
                self.model.unfreeze_alpha()
                warmup_active = False
                # Rebuild optimizer to pick up the now-trainable parameters
                self.opt = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=cfg.lr, weight_decay=cfg.weight_decay,
                )
                self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.opt,
                    T_0=cfg.scheduler_T0,
                    T_mult=cfg.scheduler_Tmult,
                    eta_min=cfg.scheduler_eta_min,
                )
                self.log(f"  >>> step {step}: warm-up complete, all params now train")

            batch = self._sample_batch(self.train_data, cfg.batch_size)
            pred = self.model(batch["q"], batch["mask"], batch["ctx"],
                              batch["m"], batch["T"], batch["r"])
            loss, components = self.loss_fn(pred, batch["C"], batch)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.sched.step()

            loss_window.append(components["data"])
            for k, v in components.items():
                comps_window.setdefault(k, deque(maxlen=cfg.log_every)).append(v)

            with torch.no_grad():
                pv = _per_episode_variance(pred, batch["tmask"]).mean().item()
                tv = _per_episode_variance(batch["C"], batch["tmask"]).mean().item()
                ratio = pv / max(tv, 1e-12)
                ratio_window.append(ratio)

            if (step + 1) % cfg.log_every == 0:
                vstats = self._eval_val()
                comps_avg = {k: float(np.mean(list(v))) for k, v in comps_window.items()}
                ratio_avg = float(np.mean(list(ratio_window)))
                with torch.no_grad():
                    alpha_mean = self.model.alpha.mean().item()
                    alpha_max  = self.model.alpha.max().item()
                    bout_v     = self.model.bout.item()
                    dead_pct   = self._dead_positive_pct(batch)
                lr_now = self.sched.get_last_lr()[0]
                comp_str = f"data {comps_avg['data']:.6f}"
                if "upper" in comps_avg:
                    comp_str += f"  upp {comps_avg['upper']:.5f}"
                if "var" in comps_avg:
                    comp_str += f"  var {comps_avg['var']:.4f}"
                self.log(
                    f"  step {step+1:5d}  {comp_str}  | "
                    f"val_MSE {vstats['val_mse']:.6f}  "
                    f"p/t var {ratio_avg:.3f}  "
                    f"bout {bout_v:+.3f}  "
                    f"alpha mean {alpha_mean:.3f} max {alpha_max:.3f}  "
                    f"dead {dead_pct:.1f}%  "
                    f"lr {lr_now:.2e}  ({time.time()-t0:.0f}s)"
                )
                self.history.append(dict(
                    step=step + 1,
                    train_data=comps_avg["data"],
                    val_mse=vstats["val_mse"],
                    var_ratio=ratio_avg,
                    bout=bout_v,
                    alpha_mean=alpha_mean,
                    dead_positive_pct=dead_pct,
                ))

        elapsed = time.time() - t0
        self.log(f"  done in {elapsed:.1f}s")
        return self.history
