"""Unit tests for RF-Diffusion runner module (src/runner.py)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestTruncateModelBlocks:
    """Tests for truncate_model_blocks utility.

    The function uses hasattr(model, "module") to detect DDP wrapping.
    Since torch.nn.Module exposes a built-in .module property, hasattr()
    is always True for any nn.Module. We must patch hasattr to treat our
    test model as if it has no .module attribute.
    """

    class _TruncatableModel(torch.nn.Module):
        """Minimal Module with a .blocks attribute for truncate_model_blocks testing."""
        def __init__(self, n_blocks: int):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(n_blocks)])
            self._module_ref = None  # not used; prevents accidental DDP unwrapping

        def forward(self, x):
            return x

    class _DDPModel(torch.nn.Module):
        """Simulates a DistributedDataParallel-wrapped model."""
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.module = inner

        def forward(self, x):
            return self.module(x)

    def test_keeps_first_k_blocks(self):
        """Truncating to k blocks should keep only the first k items."""
        from src.runner import truncate_model_blocks

        model = self._TruncatableModel(n_blocks=4)

        def _hasattr_patch(obj, name):
            if name == "module" and isinstance(obj, self._TruncatableModel):
                return False
            return object.__getattribute__(obj, "__dict__").get(name) is not None or getattr(type(obj), name, None) is not None

        with patch("src.runner.hasattr", _hasattr_patch):
            result = truncate_model_blocks(model, n_blocks=2)

        assert len(result.blocks) == 2

    def test_zero_blocks_returns_empty_module_list(self):
        """Truncating to 0 blocks should return an empty ModuleList."""
        from src.runner import truncate_model_blocks

        model = self._TruncatableModel(n_blocks=2)

        def _hasattr_patch(obj, name):
            if name == "module" and isinstance(obj, self._TruncatableModel):
                return False
            return object.__getattribute__(obj, "__dict__").get(name) is not None or getattr(type(obj), name, None) is not None

        with patch("src.runner.hasattr", _hasattr_patch):
            result = truncate_model_blocks(model, n_blocks=0)
        assert len(result.blocks) == 0

    def test_keeps_all_blocks_when_k_exceeds_length(self):
        """Requesting more blocks than exist should keep all existing blocks."""
        from src.runner import truncate_model_blocks

        model = self._TruncatableModel(n_blocks=2)

        def _hasattr_patch(obj, name):
            if name == "module" and isinstance(obj, self._TruncatableModel):
                return False
            return object.__getattribute__(obj, "__dict__").get(name) is not None or getattr(type(obj), name, None) is not None

        with patch("src.runner.hasattr", _hasattr_patch):
            result = truncate_model_blocks(model, n_blocks=10)
        assert len(result.blocks) == 2

    def test_unwraps_ddp_module(self):
        """Should unwrap DistributedDataParallel .module before truncating."""
        from src.runner import truncate_model_blocks

        inner = self._TruncatableModel(n_blocks=2)
        outer = self._DDPModel(inner)

        result = truncate_model_blocks(outer, n_blocks=1)
        assert len(result.blocks) == 1


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_defaults(self):
        """Default values should match expected project conventions."""
        from src.config import ExperimentConfig

        cfg = ExperimentConfig()

        assert cfg.task == "wifi"
        assert cfg.mode == "pretrained"
        assert cfg.seed == 11
        assert cfg.num_samples is None
        assert cfg.device == "cuda"
        assert cfg.batch_size == 1
        assert isinstance(cfg.output_dir, Path)
        assert isinstance(cfg.log_dir, Path)
        assert cfg.notes == ""

    def test_custom_values(self):
        """Passing values should override defaults."""
        from src.config import ExperimentConfig

        cfg = ExperimentConfig(task="fmcw", seed=42, num_samples=5, device="cpu")

        assert cfg.task == "fmcw"
        assert cfg.seed == 42
        assert cfg.num_samples == 5
        assert cfg.device == "cpu"

    def test_from_yaml_loads_all_fields(self, tmp_path):
        """from_yaml should load task, seed, device, and batch_size fields."""
        from src.config import ExperimentConfig
        import yaml

        data = {"task": "mimo", "seed": 99, "device": "cpu", "batch_size": 4}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(data))

        cfg = ExperimentConfig.from_yaml(path)

        assert cfg.task == "mimo"
        assert cfg.seed == 99
        assert cfg.device == "cpu"
        assert cfg.batch_size == 4


class TestBuildModel:
    """Tests for build_model (mocked, no real upstream required)."""

    def test_returns_nn_module_instance(self):
        """build_model should return a torch.nn.Module subclass instance."""
        from src.runner import build_model
        from tfdiff.params import AttrDict

        params = AttrDict({
            "task_id": 0,
            "model_dir": "/nonexistent",
        })

        device = torch.device("cpu")

        with pytest.raises((FileNotFoundError, OSError)):
            build_model(params, device)

    def test_unknown_task_id_raises(self):
        """Unknown task_id should raise ValueError."""
        from src.runner import build_model
        from tfdiff.params import AttrDict

        params = AttrDict({"task_id": 99, "model_dir": "/nonexistent"})
        device = torch.device("cpu")

        with pytest.raises(ValueError, match="Unknown task_id"):
            build_model(params, device)


class TestEvaluatePretrained:
    """Tests for evaluate_pretrained (mocked, no real upstream required)."""

    class _TruncatableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

        def forward(self, x):
            return x

    def test_returns_dict_type(self):
        """evaluate_pretrained should return a dict."""
        from src.runner import evaluate_pretrained
        from src.config import ExperimentConfig

        cfg = ExperimentConfig(task="wifi", num_samples=1, device="cpu")

        mock_model = self._TruncatableModel()

        mock_diffusion = MagicMock()
        mock_diffusion.native_sampling.return_value = torch.zeros(1, 2, dtype=torch.float32)
        mock_diffusion.fast_sampling.return_value = torch.zeros(1, 2, dtype=torch.float32)
        mock_diffusion.sampling.return_value = torch.zeros(1, 2, dtype=torch.float32)

        mock_dataset = iter([
            {
                "data": torch.zeros(1, 2, dtype=torch.float32),
                "cond": torch.zeros(1, 2, dtype=torch.float32),
            }
        ])

        with patch("src.runner.build_model", return_value=mock_model), \
             patch("src.runner.GaussianDiffusion", return_value=mock_diffusion), \
             patch("src.runner.SignalDiffusion", return_value=mock_diffusion), \
             patch("src.runner.from_path_inference", return_value=mock_dataset), \
             patch("src.runner.set_seed"), \
             patch("src.runner.reset_peak_memory"), \
             patch("src.runner.compute_ssim", return_value=0.8), \
             patch("os.makedirs"), \
             patch("src.runner.scio.savemat"):

            result = evaluate_pretrained(cfg, sampling_strategy="native")

        assert isinstance(result, dict)

    def test_result_contains_required_keys(self):
        """Result dict should contain ssim/snr metrics and count."""
        from src.runner import evaluate_pretrained
        from src.config import ExperimentConfig

        cfg = ExperimentConfig(task="wifi", num_samples=1, device="cpu")

        mock_model = self._TruncatableModel()

        mock_diffusion = MagicMock()
        mock_diffusion.native_sampling.return_value = torch.zeros(1, 2, dtype=torch.float32)

        mock_dataset = iter([
            {
                "data": torch.zeros(1, 2, dtype=torch.float32),
                "cond": torch.zeros(1, 2, dtype=torch.float32),
            }
        ])

        with patch("src.runner.build_model", return_value=mock_model), \
             patch("src.runner.GaussianDiffusion", return_value=mock_diffusion), \
             patch("src.runner.SignalDiffusion", return_value=mock_diffusion), \
             patch("src.runner.from_path_inference", return_value=mock_dataset), \
             patch("src.runner.set_seed"), \
             patch("src.runner.reset_peak_memory"), \
             patch("src.runner.compute_ssim", return_value=0.81), \
             patch("os.makedirs"), \
             patch("src.runner.scio.savemat"):

            result = evaluate_pretrained(cfg, sampling_strategy="native")

        assert "ssim" in result or "average_ssim" in result
        assert "count" in result or "num_samples" in result


class TestBuildTaskParams:
    """Tests for build_task_params utility."""

    def test_wifi_params(self):
        """WiFi task should set task_id=0 and correct model_dir suffix."""
        from src.runner import build_task_params

        params = build_task_params("wifi")

        assert params.task_id == 0
        assert "wifi" in params.model_dir
        assert "b32-256-100s" in params.model_dir

    def test_fmcw_params(self):
        """FMCW task should set task_id=1 and correct model_dir suffix."""
        from src.runner import build_task_params

        params = build_task_params("fmcw")

        assert params.task_id == 1
        assert "fmcw" in params.model_dir
        assert "b32-256-100s" in params.model_dir

    def test_mimo_params_uses_200s_suffix(self):
        """MIMO task should use 200s suffix instead of 100s."""
        from src.runner import build_task_params

        params = build_task_params("mimo")

        assert params.task_id == 2
        assert "mimo" in params.model_dir
        assert "b32-256-200s" in params.model_dir

    def test_unknown_task_raises(self):
        """Unknown task string should raise KeyError."""
        from src.runner import build_task_params

        with pytest.raises(KeyError):
            build_task_params("unknown_task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
