"""Unit tests for RF-Diffusion evaluation package."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compute_snr, aggregate_metrics


class TestEvaluationMetrics:
    """Tests for evaluation metric functions."""

    def test_aggregate_metrics_ssim(self):
        """Test SSIM aggregation."""
        ssim_list = [0.8, 0.82, 0.79, 0.81, 0.83]
        result = aggregate_metrics(ssim_list, [])

        assert "ssim" in result
        assert abs(result["ssim"]["mean"] - 0.81) < 0.01
        assert result["ssim"]["count"] == 5
        assert result["ssim"]["min"] == 0.79
        assert result["ssim"]["max"] == 0.83

    def test_aggregate_metrics_snr(self):
        """Test SNR aggregation."""
        snr_list = [28.0, 29.5, 30.1, 29.0, 30.5]
        result = aggregate_metrics([], snr_list)

        assert "snr_db" in result
        assert abs(result["snr_db"]["mean"] - 29.42) < 0.1
        assert result["snr_db"]["count"] == 5

    def test_aggregate_metrics_both(self):
        """Test combined SSIM and SNR aggregation."""
        ssim_list = [0.81, 0.82, 0.80]
        snr_list = [29.0, 30.0, 29.5]
        result = aggregate_metrics(ssim_list, snr_list)

        assert "ssim" in result
        assert "snr_db" in result

    def test_aggregate_metrics_empty(self):
        """Test empty input returns empty result."""
        result = aggregate_metrics([], [])
        assert result == {}

    def test_compute_snr(self):
        """Test SNR computation."""
        import numpy as np

        signal = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
        noise = 0.1
        pred = signal + noise * np.random.randn(3) + 1j * noise * np.random.randn(3)

        snr_db = compute_snr(pred, signal)
        assert isinstance(snr_db, float)
        assert snr_db > 0


class TestConfigModule:
    """Tests for configuration module."""

    def test_set_seed(self):
        """Test random seed setting."""
        from src.config import set_seed

        set_seed(42)
        a = torch.rand(3)
        set_seed(42)
        b = torch.rand(3)
        assert torch.allclose(a, b)

    def test_logger_creation(self):
        """Test logger creation."""
        from src.config import get_logger

        logger1 = get_logger("test")
        logger2 = get_logger("test")
        assert logger1 is logger2


class TestEvaluationModule:
    """Tests for evaluation module."""

    def test_ssim_computation_shape(self):
        """Test SSIM with the shape produced by actual callers: (2, dim)."""
        from src.evaluation import compute_ssim

        device = torch.device("cpu")
        dim = 32

        # Shape (2, dim) — channels=2 (real, imag), as runner.py produces
        pred = torch.randn(2, dim, dtype=torch.complex64)
        data = torch.randn(2, dim, dtype=torch.complex64)

        ssim = compute_ssim(pred, data, input_dim=dim, device=device)
        assert isinstance(ssim, float)
        assert -1.0 <= ssim <= 1.0

    def test_ssim_range(self):
        """Test SSIM output bounds for perfect and noisy inputs."""
        from src.evaluation import compute_ssim

        device = torch.device("cpu")
        dim = 32

        # Identical signals → SSIM should be 1
        sig = torch.randn(2, dim, dtype=torch.complex64)
        ssim_perfect = compute_ssim(sig, sig, input_dim=dim, device=device)
        assert ssim_perfect > 0.99, "SSIM of identical signals should be ~1"

        # Noisy signals → SSIM should be within valid range
        noisy = sig + 0.1 * torch.randn_like(sig)
        ssim_noisy = compute_ssim(noisy, sig, input_dim=dim, device=device)
        assert -1 <= ssim_noisy <= 1, "SSIM should be in [-1, 1]"

    def test_snr_mimo_computation(self):
        """Test MIMO SNR computation."""
        from src.evaluation import compute_snr_mimo

        pred = torch.randn(1, 8, 32, 32, 2)
        data = torch.randn(1, 8, 32, 32, 2)

        snr = compute_snr_mimo(pred, data)
        assert isinstance(snr, float)


class TestConfigYaml:
    """Tests for config YAML loading."""

    def test_experiment_config_defaults(self):
        """Test ExperimentConfig dataclass field defaults."""
        from src.config import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.task == "wifi"
        assert cfg.mode == "pretrained"
        assert cfg.seed == 11
        assert cfg.device == "cuda"
        assert cfg.batch_size == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
