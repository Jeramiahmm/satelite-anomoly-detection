"""
Synthetic Telemetry Generator.

Generates multivariate satellite telemetry data for training and simulation:
    - Voltage (V): Bus voltage with orbital eclipse transitions
    - Reaction Wheel RPM: Attitude control with periodic disturbances
    - Signal-to-Noise Ratio (dB): Communication link quality

Supports injection of attack patterns:
    - "nominal": Clean telemetry
    - "bitflip": Sporadic bit-flip errors in voltage readings
    - "spacejack": Coordinated injection across all channels
    - "polarized": Gradual polarization (subtle drift attack)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch


class AttackMode(Enum):
    NOMINAL = "nominal"
    BITFLIP = "bitflip"
    SPACEJACK = "spacejack"
    POLARIZED = "polarized"


# Nominal telemetry profiles (mean, std) per feature
NOMINAL_PROFILES = {
    "voltage_v": (28.0, 0.5),       # 28V bus typical for small LEO sats
    "reaction_wheel_rpm": (3000.0, 200.0),  # Nominal wheel speed
    "snr_db": (15.0, 2.0),          # Typical LEO-to-ground SNR
}


def generate_telemetry_batch(
    batch_size: int,
    attack_mode: AttackMode = AttackMode.NOMINAL,
    time_step: int = 0,
    attack_intensity: float = 1.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a batch of synthetic telemetry vectors.

    Args:
        batch_size: Number of telemetry samples.
        attack_mode: Type of anomaly to inject.
        time_step: Current simulation tick (for time-varying patterns).
        attack_intensity: Scaling factor for attack magnitude [0, 1].
        seed: Optional random seed for reproducibility.

    Returns:
        Tensor of shape (batch_size, 3) — [voltage, rw_rpm, snr].
    """
    if seed is not None:
        torch.manual_seed(seed)

    profiles = list(NOMINAL_PROFILES.values())
    means = torch.tensor([p[0] for p in profiles])
    stds = torch.tensor([p[1] for p in profiles])

    # Base nominal telemetry with orbital variation
    orbital_phase = 2 * math.pi * (time_step % 5400) / 5400  # ~90 min orbit
    eclipse_factor = max(0.0, math.cos(orbital_phase))  # Simulates eclipse

    telemetry = torch.randn(batch_size, 3) * stds + means

    # Add orbital variation to voltage (drops in eclipse)
    telemetry[:, 0] -= (1 - eclipse_factor) * 2.0

    # Add periodic disturbance to reaction wheel
    telemetry[:, 1] += 100 * math.sin(orbital_phase * 3)

    # Add atmospheric/path loss variation to SNR
    telemetry[:, 2] -= (1 - eclipse_factor) * 3.0

    if attack_mode == AttackMode.NOMINAL:
        return telemetry

    intensity = attack_intensity

    if attack_mode == AttackMode.BITFLIP:
        # Sporadic large spikes in voltage (simulating SEU bit-flips)
        mask = torch.rand(batch_size) < 0.3
        telemetry[mask, 0] += torch.randn(mask.sum()) * 15.0 * intensity

    elif attack_mode == AttackMode.SPACEJACK:
        # Coordinated injection: all channels pushed to extreme values
        telemetry[:, 0] += 10.0 * intensity
        telemetry[:, 1] += 2000.0 * intensity
        telemetry[:, 2] -= 10.0 * intensity

    elif attack_mode == AttackMode.POLARIZED:
        # Gradual drift — hardest to detect
        drift = min(time_step * 0.01 * intensity, 5.0)
        telemetry[:, 0] += drift
        telemetry[:, 1] += drift * 100
        telemetry[:, 2] -= drift * 0.5

    return telemetry


def normalize_telemetry(
    telemetry: torch.Tensor,
    means: Optional[torch.Tensor] = None,
    stds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Z-score normalize telemetry. Uses nominal profiles if stats not given."""
    if means is None:
        profiles = list(NOMINAL_PROFILES.values())
        means = torch.tensor([p[0] for p in profiles])
    if stds is None:
        profiles = list(NOMINAL_PROFILES.values())
        stds = torch.tensor([p[1] for p in profiles])
    return (telemetry - means) / (stds + 1e-8)
