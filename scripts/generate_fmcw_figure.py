"""Generate FMCW range-Doppler spectrogram comparison figure.

Uses the existing FMCW output mat files produced by the official inference run
or falls back to a fresh inference pass.

FMCW processing (matching upstream inference.py save_fmcw):
- Range FFT: 330-point FFT on axis=1
- Doppler FFT: 92-point FFT on axis=0
- Both are fftshift'd after FFT
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Resolve upstream root: use global upstream if the reproduction's is partial
UPSTREAM_ROOT = PROJECT_ROOT / "upstream" / "RF-Diffusion"
GLOBAL_UPSTREAM = Path("/data/zhaoshiqian/talents/talent-16/upstream/RF-Diffusion")
if not (UPSTREAM_ROOT / "model").exists() and GLOBAL_UPSTREAM.exists():
    UPSTREAM_ROOT = GLOBAL_UPSTREAM

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch

RESULTS_ROOT = PROJECT_ROOT / "results"
FIG_ROOT = PROJECT_ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def range_doppler_spec(data: np.ndarray) -> np.ndarray:
    """Compute range-Doppler spectrogram for FMCW (matches upstream save_fmcw)."""
    # data: [N, S] complex
    data = np.fft.fftshift(np.fft.fft(data, n=330, axis=1), axes=1)
    doppler = np.fft.fftshift(np.fft.fft(data, n=92, axis=0), axes=0)
    mag = np.abs(doppler)
    return 20 * np.log10(mag + 1e-6)


def load_fmcw_output():
    """Find existing FMCW output mat files from official run.
    
    The official inference.py saves to upstream dataset/fmcw/output/,
    while the wrapper saves to results/raw/fmcw/samples/.
    """
    candidates = sorted((RESULTS_ROOT / "raw").rglob("fmcw/**/sample-*.mat"))
    if candidates:
        print(f"Found {len(candidates)} FMCW output samples in results/raw")
        return candidates
    # Fallback: upstream official output directory
    upstream_output = sorted((UPSTREAM_ROOT / "dataset" / "fmcw" / "output").glob("*.mat"))
    if upstream_output:
        print(f"Found {len(upstream_output)} FMCW output samples in upstream dataset")
    return upstream_output


def run_fresh_inference() -> list[Path]:
    """Run a single-sample FMCW inference to generate comparison data."""
    print("Running fresh FMCW inference for spectrogram generation...")
    from tfdiff.params import AttrDict, all_params
    from tfdiff.fmcw_model import tfdiff_fmcw
    from tfdiff.diffusion import SignalDiffusion
    from tfdiff.dataset import from_path_inference, _nested_map

    params = all_params[1]
    model_dir = str(UPSTREAM_ROOT / "model" / "fmcw" / "b32-256-100s")
    cond_dir = str(UPSTREAM_ROOT / "dataset" / "fmcw" / "cond")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"{model_dir}/weights.pt", map_location=device)
    model = tfdiff_fmcw(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = SignalDiffusion(AttrDict(params))

    dataset = from_path_inference(AttrDict({"cond_dir": [cond_dir], **dict(params)}))

    output_dir = RESULTS_ROOT / "raw" / "fmcw_spectrogram_temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    with torch.no_grad():
        for idx, features in enumerate(dataset):
            if idx >= 3:
                break
            features = _nested_map(features,
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            data = features["data"]
            cond = features["cond"]
            pred = diffusion.native_sampling(model, data, cond, device)
            data_samples = list(torch.split(data, 1, dim=0))
            pred_samples = list(torch.split(pred, 1, dim=0))
            for b, (d_sample, p_sample) in enumerate(zip(data_samples, pred_samples)):
                d_np = torch.view_as_complex(d_sample).cpu().numpy()
                p_np = torch.view_as_complex(p_sample).cpu().numpy()
                out_path = output_dir / f"sample-{idx}-{b}.mat"
                scio.savemat(out_path, {"pred": p_np, "data": d_np})
                output_paths.append(out_path)
    return sorted(output_paths)


def fig_fmcw_spectrogram() -> Path:
    out_pdf = FIG_ROOT / "fig_fmcw_spectrogram.pdf"
    out_png = FIG_ROOT / "fig_fmcw_spectrogram.png"

    samples = load_fmcw_output()
    if not samples:
        samples = run_fresh_inference()

    # Pick the first sample for display
    sample_path = samples[0]
    print(f"Loading FMCW sample: {sample_path}")
    data = scio.loadmat(sample_path)
    pred = data["pred"]

    # The official inference.py output only contains "pred" (complex [1, N, S]).
    # Load the corresponding ground truth from the cond directory.
    cond_files = sorted(Path(UPSTREAM_ROOT / "dataset" / "fmcw" / "cond").glob("*.mat"))
    truth_raw = scio.loadmat(cond_files[0], verify_compressed_data_integrity=False)
    truth = truth_raw["feature"].astype(np.complex64)

    # Handle all possible formats:
    # - complex [1, N, S]  ← official inference.py output
    # - complex [N, S]
    # - real-view [1, N, S, 2]
    if pred.ndim == 3 and pred.dtype in (np.complex64, np.complex128):
        pred_c = pred.squeeze(0)  # [1, N, S] → [N, S]
        truth_c = truth            # ground truth is already [N, S]
    elif pred.ndim == 3 and pred.shape[-1] == 2:
        pred_c = torch.view_as_complex(torch.from_numpy(pred)).numpy()
        truth_c = torch.view_as_complex(torch.from_numpy(truth)).numpy()
    else:
        pred_c = pred
        truth_c = truth

    pred_db = range_doppler_spec(pred_c)
    truth_db = range_doppler_spec(truth_c)

    # Shared color scale
    vmin = min(truth_db.min(), pred_db.min())
    vmax = max(truth_db.max(), pred_db.max())

    fig, axes = plt.subplots(2, 1, figsize=(7, 6))

    im0 = axes[0].matshow(truth_db, cmap="viridis", origin="lower",
                           aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("Real FMCW Range-Doppler Map")
    axes[0].set_xlabel("Range bin")
    axes[0].set_ylabel("Doppler bin")
    plt.colorbar(im0, ax=axes[0], format="%+2.0f dB", fraction=0.046, pad=0.04)

    im1 = axes[1].matshow(pred_db, cmap="viridis", origin="lower",
                           aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("RF-Diffusion Generated (Reproduced)")
    axes[1].set_xlabel("Range bin")
    axes[1].set_ylabel("Doppler bin")
    plt.colorbar(im1, ax=axes[1], format="%+2.0f dB", fraction=0.046, pad=0.04)

    fig.suptitle("FMCW Radar: Real vs Generated Range-Doppler Spectrograms", fontsize=13)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[ok] saved {out_pdf} and {out_png}")
    return out_pdf


if __name__ == "__main__":
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig_fmcw_spectrogram()
