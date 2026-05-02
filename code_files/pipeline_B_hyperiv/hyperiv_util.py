"""
=============================================================================
HyperIV Model Architecture
=============================================================================
Reconstructed from Yang et al. (ICML 2025), Appendix A.

Two components:
  1. SetEmbeddingNetwork (hypernetwork g_theta): Transformer encoder that maps
     a set of 9 reference contracts Z = {(k_i, t_i, sigma_i)} to a flat
     parameter vector omega in R^337.

  2. HyperNetwork: Combines the hypernetwork with a compact MLP (iv_network)
     h_omega(k,t) -> sigma that uses omega as its weights/biases.

The iv_network architecture:
  Linear(2,16) -> Tanh -> Linear(16,16) -> Tanh -> Linear(16,1) -> Softplus
  Total parameters: (2*16+16) + (16*16+16) + (16*1+1) = 48 + 272 + 17 = 337
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 1. IV SURFACE NETWORK (compact MLP, 337 parameters)
# ============================================================
def build_iv_network():
    """
    Build the compact MLP that maps (k, t) -> implied volatility sigma.
    Architecture: 2 -> 16 (Tanh) -> 16 (Tanh) -> 1 (Softplus)
    Total: 337 parameters.
    """
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
        nn.Softplus(),
    )


def count_iv_params():
    """Return the number of parameters in the IV network."""
    net = build_iv_network()
    return sum(p.numel() for p in net.parameters())


# ============================================================
# 2. SET EMBEDDING NETWORK (Hypernetwork)
# ============================================================
class SetEmbeddingNetwork(nn.Module):
    """
    Transformer-based set embedding network (hypernetwork).

    Maps a set of reference contracts Z = {(k_i, t_i, sigma_i)}_{i=1}^M
    to a parameter vector omega in R^P.

    Architecture:
      1. FC: input_dim -> hidden_dim  (per-element embedding)
      2. Transformer encoder layers (cross-element attention)
      3. Mean pooling over set dimension
      4. FC: hidden_dim -> output_dim (parameter vector)

    No positional embeddings -> permutation invariant.

    Args:
        input_dim: features per reference contract (default 3: k, t, sigma)
        output_dim: number of parameters to generate (337 for the IV network)
        num_heads: number of attention heads
        num_layers: number of transformer encoder layers
        hidden_dim: hidden dimension throughout
    """

    def __init__(self, input_dim=3, output_dim=337, num_heads=2,
                 num_layers=2, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=0,
                activation="relu",
            )
            for _ in range(num_layers)
        ])
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, M, input_dim) — set of M reference contracts

        Returns:
            omega: (batch_size, output_dim) — parameter vector
        """
        x = self.fc1(x)
        for layer in self.attention_layers:
            x = layer(x)
        x = x.mean(dim=1)  # Mean pooling over set elements
        x = self.fc2(x)
        return x


# ============================================================
# 3. HYPER NETWORK (combines hypernetwork + IV network)
# ============================================================
class HyperNetwork(nn.Module):
    """
    Full HyperIV model: hypernetwork generates weights for the IV network.

    Forward pass:
      1. omega = g_theta(Z)           — hypernetwork produces weight vector
      2. sigma = h_omega(k, t)        — IV network evaluates surface

    The IV network's parameters are NOT trained directly; they are generated
    by the hypernetwork on-the-fly for each input reference set Z.

    Args:
        hyper_net: SetEmbeddingNetwork instance
        iv_network: nn.Sequential — the compact MLP template (for shape info)
    """

    def __init__(self, hyper_net, iv_network):
        super().__init__()
        self.hyper_net = hyper_net

        # Extract the shapes of each parameter in the IV network
        self.param_shapes = []
        self.param_names = []
        for name, param in iv_network.named_parameters():
            self.param_shapes.append(param.shape)
            self.param_names.append(name)

        # Verify total params match hypernetwork output
        total = sum(np.prod(s) for s in self.param_shapes)
        assert total == hyper_net.fc2.out_features, \
            f"IV network has {total} params but hypernetwork outputs {hyper_net.fc2.out_features}"

        # Store iv_network architecture info for reconstruction
        self._iv_template = iv_network

    def forward(self, z_batch, kt_batch):
        """
        Args:
            z_batch: (B, M, 3) — reference sets (M contracts x 3 features each)
            kt_batch: (B, N, 2) — query points (k, t) to evaluate

        Returns:
            sigma: (B, N, 1) — predicted implied volatilities
        """
        # Generate weight vector
        omega = self.hyper_net(z_batch)  # (B, P)

        # Split omega into individual parameter tensors
        params = self._split_params(omega)

        # Apply IV network with generated weights (batched)
        return self._functional_forward(kt_batch, params)

    def get_weights(self, z_batch):
        """
        Extract just the weight vector omega for a batch of reference sets.

        Args:
            z_batch: (B, M, 3) — reference sets

        Returns:
            omega: (B, P) — weight vectors (P=337)
        """
        return self.hyper_net(z_batch)

    def forward_from_weights(self, omega, kt_batch):
        """
        Evaluate IV network using pre-computed weight vectors.

        Args:
            omega: (B, P) — weight vectors
            kt_batch: (B, N, 2) — query points

        Returns:
            sigma: (B, N, 1) — predicted implied volatilities
        """
        params = self._split_params(omega)
        return self._functional_forward(kt_batch, params)

    def _split_params(self, omega):
        """Split flat parameter vector into list of tensors matching IV network shapes."""
        params = []
        offset = 0
        for shape in self.param_shapes:
            numel = int(np.prod(shape))
            param = omega[:, offset:offset + numel].reshape(-1, *shape)
            params.append(param)
            offset += numel
        return params

    def _functional_forward(self, x, params):
        """
        Functional forward pass through the IV network using generated weights.

        The IV network is: Linear -> Tanh -> Linear -> Tanh -> Linear -> Softplus
        Parameters order: [w0, b0, w1, b1, w2, b2]

        Args:
            x: (B, N, 2) — input (k, t) pairs
            params: list of parameter tensors from _split_params

        Returns:
            (B, N, 1) — predicted sigma values
        """
        # params = [weight0, bias0, weight1, bias1, weight2, bias2]
        # Layer 0: Linear(2, 16) + Tanh
        w0, b0 = params[0], params[1]  # w0: (B, 16, 2), b0: (B, 16)
        x = torch.bmm(x, w0.transpose(1, 2)) + b0.unsqueeze(1)
        x = torch.tanh(x)

        # Layer 1: Linear(16, 16) + Tanh
        w1, b1 = params[2], params[3]  # w1: (B, 16, 16), b1: (B, 16)
        x = torch.bmm(x, w1.transpose(1, 2)) + b1.unsqueeze(1)
        x = torch.tanh(x)

        # Layer 2: Linear(16, 1) + Softplus
        w2, b2 = params[4], params[5]  # w2: (B, 1, 16), b2: (B, 1)
        x = torch.bmm(x, w2.transpose(1, 2)) + b2.unsqueeze(1)
        x = torch.nn.functional.softplus(x)

        return x


# ============================================================
# 4. CONVENIENCE FUNCTIONS
# ============================================================
def create_hyperiv_model(input_dim=3, hidden_dim=128, num_heads=2,
                         num_layers=2, device=None):
    """
    Create a complete HyperIV model.

    Returns:
        model: HyperNetwork instance
        iv_network: the template IV network (for reference)
    """
    iv_network = build_iv_network()
    n_params = sum(p.numel() for p in iv_network.parameters())

    hyper_net = SetEmbeddingNetwork(
        input_dim=input_dim,
        output_dim=n_params,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

    model = HyperNetwork(hyper_net, iv_network)

    if device is not None:
        model = model.to(device)

    return model, iv_network


def assign_weights_to_iv_network(iv_network, omega):
    """
    Assign a weight vector omega to an IV network instance.

    Args:
        iv_network: nn.Sequential — the IV network
        omega: 1D tensor of shape (337,)

    Returns:
        iv_network with updated parameters (in-place)
    """
    offset = 0
    with torch.no_grad():
        for param in iv_network.parameters():
            numel = param.numel()
            param.copy_(omega[offset:offset + numel].reshape(param.shape))
            offset += numel
    return iv_network


if __name__ == "__main__":
    # Quick test
    model, iv_net = create_hyperiv_model()
    n_model_params = sum(p.numel() for p in model.parameters())
    n_iv_params = sum(p.numel() for p in iv_net.parameters())
    print(f"IV network parameters: {n_iv_params}")
    print(f"HyperIV model parameters: {n_model_params:,}")

    # Test forward pass
    B, M, N = 4, 9, 100
    z = torch.randn(B, M, 3)
    kt = torch.randn(B, N, 2)
    sigma = model(z, kt)
    print(f"Input ref set: {z.shape}, Query points: {kt.shape}")
    print(f"Output sigma: {sigma.shape}")
    print(f"Sigma range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")

    # Test weight extraction
    omega = model.get_weights(z)
    print(f"Weight vector shape: {omega.shape}")
