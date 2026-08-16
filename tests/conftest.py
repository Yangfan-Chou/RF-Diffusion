"""Pytest configuration: mock tfdiff upstream so runner.py imports succeed without real weights."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"

if "tfdiff" not in sys.modules:
    tfdiff_pkg = MagicMock()
    tfdiff_pkg.__path__ = [str(UPSTREAM_ROOT)]
    sys.modules["tfdiff"] = tfdiff_pkg

    for sub in ("dataset", "diffusion", "eeg_model", "fmcw_model",
                "mimo_model", "params", "wifi_model"):
        sys.modules[f"tfdiff.{sub}"] = MagicMock()

    class AttrDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

        def override(self, other):
            self.update(other)
            return self

    sys.modules["tfdiff.params"].AttrDict = AttrDict
    sys.modules["tfdiff.params"].all_params = [
        AttrDict({"signal_diffusion": False, "input_dim": 512}),
        AttrDict({"signal_diffusion": False, "input_dim": 512}),
        AttrDict({"signal_diffusion": True, "input_dim": 512}),
        AttrDict({"signal_diffusion": False, "input_dim": 512}),
    ]

    sys.modules["tfdiff.wifi_model"].tfdiff_WiFi = MagicMock()
    sys.modules["tfdiff.fmcw_model"].tfdiff_fmcw = MagicMock()
    sys.modules["tfdiff.mimo_model"].tfdiff_mimo = MagicMock()
    sys.modules["tfdiff.eeg_model"].tfdiff_eeg = MagicMock()

    def _nested_map(data, fn):
        if isinstance(data, dict):
            return {k: _nested_map(v, fn) for k, v in data.items()}
        return fn(data)

    sys.modules["tfdiff.dataset"]._nested_map = _nested_map
    sys.modules["tfdiff.dataset"].from_path_inference = MagicMock()

    sys.modules["tfdiff.diffusion"].GaussianDiffusion = MagicMock()
    sys.modules["tfdiff.diffusion"].SignalDiffusion = MagicMock()
