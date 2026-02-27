"""
Variational Autoencoder (VAE) for Satellite Telemetry Anomaly Detection.

The VAE is trained on multivariate telemetry vectors:
    [Voltage (V), Reaction Wheel RPM, Signal-to-Noise Ratio (dB)]

Anomalies are detected via the reconstruction error: if the input cannot be
faithfully reconstructed, the telemetry is deemed anomalous — indicative of
"Space-Jacking" or "Bit-Flip" attacks (per CCSDS 350.0-G-3 threat model).

Architecture:
    Encoder: input_dim → 64 → 32 → (μ, log σ²)   [latent_dim=8]
    Decoder: latent_dim → 32 → 64 → input_dim

Loss: reconstruction (MSE) + KL divergence
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Telemetry feature names for documentation and logging
TELEMETRY_FEATURES = ["voltage_v", "reaction_wheel_rpm", "snr_db"]
INPUT_DIM = len(TELEMETRY_FEATURES)
LATENT_DIM = 8


class TelemetryVAE(nn.Module):
    """
    Variational Autoencoder for multivariate satellite telemetry.

    Forward pass returns (reconstruction, mu, log_var) for loss computation.
    The `reconstruct` method returns only the reconstruction for inference.
    """

    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, 64)
        self.enc_fc2 = nn.Linear(64, 32)
        self.enc_mu = nn.Linear(32, latent_dim)
        self.enc_logvar = nn.Linear(32, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 32)
        self.dec_fc2 = nn.Linear(32, 64)
        self.dec_out = nn.Linear(64, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters (μ, log σ²)."""
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to telemetry space."""
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            (x_reconstructed, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only reconstruction (no gradient, deterministic)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            return self.decode(mu)  # Use mean directly (no sampling)


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 0.5,
) -> torch.Tensor:
    """
    VAE loss = Reconstruction (MSE) + β * KL Divergence.

    The KL weight (β) balances reconstruction fidelity against latent
    regularization. For anomaly detection we favor reconstruction accuracy,
    so β is set conservatively.
    """
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_weight * kl_loss


def reconstruction_error(model: TelemetryVAE, x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample reconstruction error (L2 norm squared).

    This is the anomaly score: higher values indicate more anomalous
    telemetry. Returns a 1-D tensor of shape (batch_size,).
    """
    x_hat = model.reconstruct(x)
    return torch.sum((x - x_hat) ** 2, dim=-1)
