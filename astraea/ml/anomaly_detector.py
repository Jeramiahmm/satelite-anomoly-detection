"""
Dynamic Anomaly Detection Engine.

Uses an Exponential Moving Average (EMA) of the VAE reconstruction error
to maintain a dynamic threshold τ. This accounts for the naturally varying
telemetry profiles as the satellite orbits (eclipse transitions, ground
station handoffs, etc.).

Threshold policy:
    τ = μ_ema + k * σ_ema

Where k is a configurable sensitivity multiplier (default 3.0, per the
"3-sigma rule" adapted for non-Gaussian distributions common in space
telemetry).

Key design: The EMA is ONLY updated when the current score does not exceed
the threshold. This prevents attack data from polluting the baseline and
allows the detector to maintain a stable reference during active attacks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from .vae import TelemetryVAE, reconstruction_error

logger = logging.getLogger(__name__)


@dataclass
class AnomalyState:
    """Current state of the anomaly detector for a single node."""

    ema_mean: float = 0.0
    ema_var: float = 1.0
    threshold: float = float("inf")
    last_score: float = 0.0
    last_normalized_score: float = 0.0
    anomaly_count: int = 0
    sample_count: int = 0
    is_anomalous: bool = False
    detection_timestamp: float = 0.0


class AnomalyDetector:
    """
    Real-time anomaly detector wrapping a TelemetryVAE.

    The detector maintains a rolling EMA of the reconstruction error
    distribution and flags samples that exceed the dynamic threshold.

    Critically, the EMA baseline is only updated with samples that fall
    within the current threshold. This prevents adversarial telemetry
    from shifting the baseline and evading detection.
    """

    def __init__(
        self,
        model: TelemetryVAE,
        ema_alpha: float = 0.05,
        sigma_multiplier: float = 3.0,
        warmup_samples: int = 50,
        anomaly_ceiling: float = 0.8,
    ):
        """
        Args:
            model: Trained TelemetryVAE instance.
            ema_alpha: EMA smoothing factor (lower = more inertia).
            sigma_multiplier: Number of sigma for threshold (k).
            warmup_samples: Samples before threshold becomes active.
            anomaly_ceiling: Normalized score above which → anomaly flag.
        """
        self.model = model
        self.alpha = ema_alpha
        self.k = sigma_multiplier
        self.warmup = warmup_samples
        self.ceiling = anomaly_ceiling
        self.state = AnomalyState()
        self._baseline_frozen = False

    def ingest(self, telemetry: torch.Tensor) -> AnomalyState:
        """
        Ingest a telemetry vector and return the updated anomaly state.

        Args:
            telemetry: Tensor of shape (input_dim,) or (1, input_dim).

        Returns:
            Updated AnomalyState with score and anomaly flag.
        """
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)

        raw_score = reconstruction_error(self.model, telemetry).item()
        self.state.sample_count += 1
        self.state.last_score = raw_score

        in_warmup = self.state.sample_count <= self.warmup

        # During warmup: always update EMA to establish baseline
        # After warmup: only update if score is within threshold (not anomalous)
        should_update_ema = in_warmup or raw_score <= self.state.threshold

        if should_update_ema:
            if self.state.sample_count == 1:
                self.state.ema_mean = raw_score
                self.state.ema_var = 0.0
            else:
                delta = raw_score - self.state.ema_mean
                self.state.ema_mean += self.alpha * delta
                self.state.ema_var = (1 - self.alpha) * (
                    self.state.ema_var + self.alpha * delta**2
                )

        # Compute dynamic threshold: τ = μ + k * σ
        ema_std = max(self.state.ema_var**0.5, 1e-6)
        self.state.threshold = self.state.ema_mean + self.k * ema_std

        # Normalize score to [0, 1] range for policy evaluation
        if in_warmup:
            self.state.last_normalized_score = 0.0
            self.state.is_anomalous = False
        else:
            # Sigmoid-like normalization centered on threshold
            excess = (raw_score - self.state.threshold) / max(ema_std, 1e-6)
            normalized = 1.0 / (1.0 + torch.exp(torch.tensor(-excess)).item())
            self.state.last_normalized_score = normalized

            was_anomalous = self.state.is_anomalous
            self.state.is_anomalous = normalized > self.ceiling

            if self.state.is_anomalous:
                self.state.anomaly_count += 1
                if not was_anomalous:
                    self.state.detection_timestamp = time.time()
                    logger.warning(
                        "ANOMALY DETECTED: raw_score=%.4f  threshold=%.4f  "
                        "normalized=%.4f  count=%d",
                        raw_score,
                        self.state.threshold,
                        normalized,
                        self.state.anomaly_count,
                    )

        return self.state

    def get_anomaly_score(self) -> float:
        """Return the current normalized anomaly score [0, 1]."""
        return self.state.last_normalized_score

    def reset_ema(self) -> None:
        """Reset EMA statistics (e.g., after model update from federation)."""
        self.state.ema_mean = 0.0
        self.state.ema_var = 1.0
        self.state.sample_count = 0
        self.state.threshold = float("inf")
        logger.info("EMA statistics reset (post-federation model update)")
