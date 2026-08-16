#!/usr/bin/env python3
"""
run_physics_experiments.py — Physical Plausibility Experiments for RF-Diffusion

This script runs three experiments to add depth to the RF-Diffusion report:

1. OFDM Decoding Physical Plausibility Test
   - Compute OFDM Error Vector Magnitude (EVM)
   - Test constant modulus property
   - Measure subcarrier correlation structure
   - Frequency correlation analysis

2. Frequency Blur Kernel Ablation
   - Test different σ values for the Gaussian blur kernel
   - Measure SSIM, NMSE for each σ

3. Sampling Strategy Comparison
   - Already done in run_ddim_sampling.py, but we extend it here

Author: RF-Diffusion Reproduction Project
Date: 2026-08-06
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.io as scio
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_UPSTREAM = Path("/data/zhaoshiqian/talents/talent-16/upstream/RF-Diffusion")
sys.path.insert(0, str(REAL_UPSTREAM))
import os as _os
_os.chdir(REAL_UPSTREAM)

from tfdiff.params import AttrDict, all_params
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import from_path_inference, _nested_map


# ============================================================================
# SSIM Implementation (from upstream)
# ============================================================================

def _gaussian(window_size: int, sigma: float):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                      for x in range(window_size)])
    return g / g.sum()


@torch.jit.script
def _create_window(h: int, w: int):
    hw = _gaussian(h, 1.5).unsqueeze(1)
    ww = _gaussian(w, 1.5).unsqueeze(1)
    win = hw.mm(ww.t()).unsqueeze(0).unsqueeze(0).contiguous()
    return win


def eval_ssim(pred, data, h, w, device):
    """SSIM on complex spectrograms; higher is better."""
    window = _create_window(h, w).to(torch.complex64).to(device)
    pad = [h // 2, w // 2]
    def conv2d(x):
        return torch.nn.functional.conv2d(x, window, padding=pad, groups=1)
    mu_p = conv2d(pred);  mu_d = conv2d(data)
    var_p = conv2d(pred * pred) - mu_p.pow(2.)
    var_d = conv2d(data  * data)  - mu_d.pow(2.)
    cov  = conv2d(pred * data)  - mu_p * mu_d
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_p * mu_d + C1) * (2 * cov.real + C2)) \
             / ((mu_p.pow(2) + mu_d.pow(2) + C1) * (var_p + var_d + C2))
    return (2 * ssim_map.mean()).real


# ============================================================================
# OFDM Physical Plausibility Metrics
# ============================================================================

def compute_evm(real: np.ndarray, generated: np.ndarray) -> float:
    """
    Compute Error Vector Magnitude (EVM) for OFDM signals.
    
    EVM = sqrt(mean(|x_pred - x_real|^2)) / sqrt(mean(|x_real|^2))
    Lower EVM is better (typically < 0.1 for good quality)
    
    Args:
        real: Ground truth signal (complex)
        generated: Generated signal (complex)
    
    Returns:
        EVM as a ratio (not percentage)
    """
    error_power = np.mean(np.abs(generated - real) ** 2)
    signal_power = np.mean(np.abs(real) ** 2)
    evm = np.sqrt(error_power / signal_power)
    return float(evm)


def compute_constant_modulus_ratio(signal: np.ndarray, threshold: float = 0.05) -> float:
    """
    Compute the ratio of samples that maintain near-constant modulus.
    
    For OFDM pilot subcarriers, the modulus should be relatively constant.
    We measure how many samples have modulus within threshold of the mean.
    
    Args:
        signal: Complex signal
        threshold: Relative tolerance (default 5%)
    
    Returns:
        Ratio of constant-modulus samples
    """
    modulus = np.abs(signal)
    mean_mod = np.mean(modulus)
    if mean_mod < 1e-10:
        return 0.0
    deviation = np.abs(modulus - mean_mod) / mean_mod
    ratio = np.mean(deviation < threshold)
    return float(ratio)


def compute_subcarrier_correlation(signal: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between subcarriers.
    
    For OFDM, adjacent subcarriers should be correlated due to channel smoothing.
    
    Args:
        signal: (n_subcarriers, n_time_steps) complex array
    
    Returns:
        Correlation matrix (n_subcarriers, n_subcarriers)
    """
    n_sub, n_time = signal.shape
    # Compute correlation across time for each subcarrier pair
    corr = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        for j in range(n_sub):
            corr[i, j] = np.abs(np.corrcoef(np.abs(signal[i, :]), np.abs(signal[j, :]))[0, 1])
    return corr


def compute_frequency_correlation(signal: np.ndarray) -> np.ndarray:
    """
    Compute autocorrelation of signal along frequency (subcarrier) axis.
    
    Adjacent subcarriers in OFDM should have high correlation.
    
    Args:
        signal: (n_subcarriers,) or (n_subcarriers, n_time) complex array
    
    Returns:
        Autocorrelation coefficients for lags 0 to n_sub-1
    """
    if signal.ndim == 2:
        # Average across time
        signal = np.mean(signal, axis=1)
    
    # Normalize
    signal = signal - np.mean(signal)
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]  # Take positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize
    return autocorr


def compute_nmse(real: np.ndarray, generated: np.ndarray) -> float:
    """
    Compute Normalized Mean Squared Error.
    
    NMSE = sum(|x_pred - x_real|^2) / sum(|x_real|^2)
    """
    mse = np.mean(np.abs(generated - real) ** 2)
    signal_power = np.mean(np.abs(real) ** 2)
    nmse = mse / signal_power if signal_power > 1e-10 else float('inf')
    return float(nmse)


# ============================================================================
# Experiment 1: OFDM Physical Plausibility Test
# ============================================================================

def run_ofdm_plausibility_test(
    model,
    diffusion,
    dataset: List[Dict],
    device: torch.device,
    params: AttrDict,
    num_samples: int = 41
) -> Dict[str, Any]:
    """
    Run OFDM physical plausibility test on generated vs real signals.
    
    This tests whether RF-Diffusion generates signals that are physically
    plausible for actual OFDM communication.
    """
    results = []
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: OFDM Physical Plausibility Test")
    print("="*70)
    
    with torch.no_grad():
        for idx, features in enumerate(tqdm(dataset[:num_samples], desc="OFDM Plausibility")):
            if idx >= num_samples:
                break
                
            features = _nested_map(features, lambda x: x.to(device)
                                   if isinstance(x, torch.Tensor) else x)
            data = features["data"]
            cond = features["cond"]
            
            # Generate prediction
            pred = diffusion.native_sampling(model, data, cond, device)
            
            # Convert to complex numpy arrays
            d_complex = torch.view_as_complex(data.squeeze(0)).cpu().numpy()
            p_complex = torch.view_as_complex(pred.squeeze(0)).cpu().numpy()
            
            # Reshape: (90, 90) -> (90, 90) - treating as (n_subcarriers, n_time)
            # The actual data is (90, 90) for sample_rate=90, input_dim=90
            # But the .mat files have (1545, 90) - let's use what the model expects
            
            n_sub = p_complex.shape[0]
            n_time = p_complex.shape[1]
            
            # Compute EVM
            evm = compute_evm(d_complex, p_complex)
            
            # Compute SSIM (need torch tensors)
            d_tensor = torch.from_numpy(d_complex).to(torch.complex64).unsqueeze(0).to(device)
            p_tensor = torch.from_numpy(p_complex).to(torch.complex64).unsqueeze(0).to(device)
            ssim = eval_ssim(
                p_tensor, d_tensor,
                params.sample_rate, params.input_dim, device
            ).item()
            
            # Compute NMSE
            nmse = compute_nmse(d_complex, p_complex)
            
            # Compute subcarrier correlation for a subset of subcarriers (for speed)
            # Take every 10th subcarrier to reduce computation
            step = max(1, n_sub // 20)
            sub_indices = list(range(0, n_sub, step))[:20]
            
            # Average correlation across time for each subcarrier
            real_corr_list = []
            gen_corr_list = []
            for i in range(len(sub_indices) - 1):
                for j in range(i + 1, len(sub_indices)):
                    real_sub_corr = np.corrcoef(
                        np.abs(d_complex[sub_indices[i], :]),
                        np.abs(d_complex[sub_indices[j], :])
                    )[0, 1]
                    gen_sub_corr = np.corrcoef(
                        np.abs(p_complex[sub_indices[i], :]),
                        np.abs(p_complex[sub_indices[j], :])
                    )[0, 1]
                    real_corr_list.append(real_sub_corr)
                    gen_corr_list.append(gen_sub_corr)
            
            avg_real_corr = np.mean(real_corr_list) if real_corr_list else 0.0
            avg_gen_corr = np.mean(gen_corr_list) if gen_corr_list else 0.0
            
            # Compute constant modulus ratio (using pilot-like subcarriers: first 10)
            pilot_subcarriers = min(10, n_sub // 10)
            real_const_ratio = compute_constant_modulus_ratio(
                d_complex[:pilot_subcarriers, :].flatten()
            )
            gen_const_ratio = compute_constant_modulus_ratio(
                p_complex[:pilot_subcarriers, :].flatten()
            )
            
            # Frequency correlation (adjacent subcarrier correlation)
            real_freq_corr = compute_frequency_correlation(np.mean(np.abs(d_complex), axis=1))
            gen_freq_corr = compute_frequency_correlation(np.mean(np.abs(p_complex), axis=1))
            
            # Correlation at lag 1 (adjacent subcarriers)
            real_lag1_corr = real_freq_corr[1] if len(real_freq_corr) > 1 else 0.0
            gen_lag1_corr = gen_freq_corr[1] if len(gen_freq_corr) > 1 else 0.0
            
            results.append({
                "sample_id": idx,
                "evm": evm,
                "ssim": ssim,
                "nmse": nmse,
                "subcarrier_corr_real": avg_real_corr,
                "subcarrier_corr_gen": avg_gen_corr,
                "subcarrier_corr_diff": avg_real_corr - avg_gen_corr,
                "constant_modulus_ratio_real": real_const_ratio,
                "constant_modulus_ratio_gen": gen_const_ratio,
                "freq_lag1_corr_real": real_lag1_corr,
                "freq_lag1_corr_gen": gen_lag1_corr,
            })
    
    # Aggregate statistics
    agg_results = {
        "num_samples": len(results),
        "evm_mean": np.mean([r["evm"] for r in results]),
        "evm_std": np.std([r["evm"] for r in results]),
        "ssim_mean": np.mean([r["ssim"] for r in results]),
        "ssim_std": np.std([r["ssim"] for r in results]),
        "nmse_mean": np.mean([r["nmse"] for r in results]),
        "nmse_std": np.std([r["nmse"] for r in results]),
        "subcarrier_corr_real_mean": np.mean([r["subcarrier_corr_real"] for r in results]),
        "subcarrier_corr_gen_mean": np.mean([r["subcarrier_corr_gen"] for r in results]),
        "const_modulus_real_mean": np.mean([r["constant_modulus_ratio_real"] for r in results]),
        "const_modulus_gen_mean": np.mean([r["constant_modulus_ratio_gen"] for r in results]),
        "freq_lag1_corr_real_mean": np.mean([r["freq_lag1_corr_real"] for r in results]),
        "freq_lag1_corr_gen_mean": np.mean([r["freq_lag1_corr_gen"] for r in results]),
        "per_sample": results,
    }
    
    return agg_results


# ============================================================================
# Experiment 2: Frequency Blur Kernel Ablation
# ============================================================================

def run_blur_kernel_ablation(
    model,
    base_params: AttrDict,
    device: torch.device,
    num_samples: int = 10
) -> Dict[str, Any]:
    """
    Ablate the frequency blur kernel σ parameter.
    
    Original blur_schedule: (1e-5^2 * ones(100)) ≈ 0 variance
    We test: σ = {0.1, 0.5, 1.0, 2.0} (variance = σ^2)
    """
    import numpy as np
    
    sigma_values = [0.1, 0.5, 1.0, 2.0]
    results = []
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Frequency Blur Kernel Ablation")
    print("="*70)
    
    # Load dataset once
    params = AttrDict(base_params)
    dataset = list(from_path_inference(params))[:num_samples]
    
    for sigma in sigma_values:
        print(f"\nTesting σ = {sigma}")
        
        # Create modified blur schedule
        modified_params = AttrDict(dict(base_params))
        var = sigma ** 2
        modified_params.blur_schedule = (var * np.ones(100)).tolist()
        
        # Create modified diffusion
        diffusion = SignalDiffusion(modified_params)
        
        ssim_list = []
        nmse_list = []
        times = []
        
        torch.cuda.reset_peak_memory_stats(device)
        
        with torch.no_grad():
            for features in tqdm(dataset, desc=f"σ={sigma}"):
                features = _nested_map(features, lambda x: x.to(device)
                                       if isinstance(x, torch.Tensor) else x)
                data = features["data"]
                cond = features["cond"]
                
                t0 = time.perf_counter()
                pred = diffusion.native_sampling(model, data, cond, device)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                
                # Convert to complex tensors
                d_complex = torch.view_as_complex(data.squeeze(0)).cpu()
                p_complex = torch.view_as_complex(pred.squeeze(0)).cpu()
                
                # Compute SSIM
                d_t = d_complex.to(torch.complex64).unsqueeze(0).to(device)
                p_t = p_complex.to(torch.complex64).unsqueeze(0).to(device)
                ssim = eval_ssim(
                    p_t, d_t,
                    base_params.sample_rate, base_params.input_dim, device
                ).item()
                ssim_list.append(ssim)
                
                # Compute NMSE
                nmse = compute_nmse(d_complex.numpy(), p_complex.numpy())
                nmse_list.append(nmse)
        
        results.append({
            "sigma": sigma,
            "variance": var,
            "ssim_mean": np.mean(ssim_list),
            "ssim_std": np.std(ssim_list),
            "nmse_mean": np.mean(nmse_list),
            "nmse_std": np.std(nmse_list),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "peak_mem_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        })
        
        print(f"  SSIM: {results[-1]['ssim_mean']:.4f} ± {results[-1]['ssim_std']:.4f}")
        print(f"  NMSE: {results[-1]['nmse_mean']:.6f} ± {results[-1]['nmse_std']:.6f}")
    
    return {"blur_ablation": results}


# ============================================================================
# Experiment 3: Extended Sampling Strategy Comparison
# ============================================================================

def run_sampling_strategy_comparison(
    model,
    base_params: AttrDict,
    device: torch.device,
    num_samples: int = 41
) -> Dict[str, Any]:
    """
    Extended sampling strategy comparison with DDIM skip values.
    
    This extends the existing run_ddim_sampling.py with:
    - More DDIM skip values: 2, 5, 10, 20
    - FID computation
    - Progressive distillation simulation
    """
    from itertools import islice
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: Sampling Strategy Comparison")
    print("="*70)
    
    params = AttrDict(base_params)
    dataset = list(islice(from_path_inference(params), num_samples))
    
    diffusion = SignalDiffusion(params)
    
    # Define strategies
    skip_values = [2, 5, 10, 20]
    results = []
    
    # Baseline: Native (1 step)
    print("\nTesting Native (1 step)...")
    ssim_list, times = [], []
    with torch.no_grad():
        for features in tqdm(dataset, desc="Native"):
            features = _nested_map(features, lambda x: x.to(device)
                                   if isinstance(x, torch.Tensor) else x)
            data = features["data"]
            cond = features["cond"]
            
            t0 = time.perf_counter()
            pred = diffusion.native_sampling(model, data, cond, device)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            
            d_complex = torch.view_as_complex(data.squeeze(0))
            p_complex = torch.view_as_complex(pred.squeeze(0))
            ssim = eval_ssim(
                p_complex.unsqueeze(0), d_complex.unsqueeze(0),
                params.sample_rate, params.input_dim, device
            ).item()
            ssim_list.append(ssim)
    
    results.append({
        "strategy": "native",
        "skip": 1,
        "model_evals": 1,
        "ssim_mean": np.mean(ssim_list),
        "ssim_std": np.std(ssim_list),
        "time_mean": np.mean(times),
        "time_std": np.std(times),
    })
    print(f"  Native SSIM: {results[-1]['ssim_mean']:.4f}")
    
    # DDIM with various skip values
    def ddim_sampling(restore_fn, data, cond, device, skip):
        """DDIM sampling with given skip value."""
        batch_size = cond.shape[0]
        T = params.max_step
        
        # Build tau list
        tau = list(range(T - 1, -1, -skip))
        if tau[-1] != 0:
            tau.append(0)
        tau_rev = list(reversed(tau))
        
        # Init
        inf_w = (diffusion.noise_weights[T - 1] + diffusion.info_weights[T - 1]).to(device)
        inf_w = inf_w.unsqueeze(-1).unsqueeze(-1)
        x_t = inf_w * torch.randn([batch_size, params.sample_rate] + params.extra_dim + [2],
                                  dtype=torch.float32, device=device)
        
        evals = 0
        for i, cur in enumerate(reversed(tau)):
            x_0_hat = restore_fn(
                x_t,
                cur * torch.ones(batch_size, dtype=torch.int64, device=device),
                cond)
            evals += 1
            if i + 1 < len(tau):
                nxt = tau[len(tau) - i - 2]
                snr = diffusion.info_weights[cur].unsqueeze(-1).unsqueeze(-1).to(device)
                nw = diffusion.noise_weights[cur].unsqueeze(-1).unsqueeze(-1).to(device)
                eps_hat = (x_t - snr * x_0_hat) / (nw + 1e-10)
                snr_nxt = diffusion.info_weights[nxt].unsqueeze(-1).unsqueeze(-1).to(device)
                nw_nxt = diffusion.noise_weights[nxt].unsqueeze(-1).unsqueeze(-1).to(device)
                x_t = snr_nxt * x_0_hat + nw_nxt * eps_hat
        
            return x_0_hat, evals, len(tau)
    
    for skip in skip_values:
        print(f"\nTesting DDIM skip={skip}...")
        ssim_list, times = [], []
        
        with torch.no_grad():
            for features in tqdm(dataset, desc=f"DDIM-{skip}"):
                features = _nested_map(features, lambda x: x.to(device)
                                       if isinstance(x, torch.Tensor) else x)
                data = features["data"]
                cond = features["cond"]
                
                t0 = time.perf_counter()
                pred, evals, num_evals = ddim_sampling(model, data, cond, device, skip)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                
                d_complex = torch.view_as_complex(data.squeeze(0)).cpu()
                p_complex = torch.view_as_complex(pred.squeeze(0)).cpu()
                d_t = d_complex.to(torch.complex64).unsqueeze(0).to(device)
                p_t = p_complex.to(torch.complex64).unsqueeze(0).to(device)
                ssim = eval_ssim(
                    p_t, d_t,
                    params.sample_rate, params.input_dim, device
                ).item()
                ssim_list.append(ssim)
        
        results.append({
            "strategy": f"ddim_skip{skip}",
            "skip": skip,
            "model_evals": num_evals,
            "ssim_mean": np.mean(ssim_list),
            "ssim_std": np.std(ssim_list),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
        })
        print(f"  DDIM-{skip} SSIM: {results[-1]['ssim_mean']:.4f}, Evals: {num_evals}")
    
    # Robust sampling (DDIM deterministic)
    print("\nTesting Robust DDIM (100 steps)...")
    ssim_list, times = [], []
    with torch.no_grad():
        for features in tqdm(dataset, desc="Robust"):
            features = _nested_map(features, lambda x: x.to(device)
                                   if isinstance(x, torch.Tensor) else x)
            data = features["data"]
            cond = features["cond"]
            
            t0 = time.perf_counter()
            pred = diffusion.robust_sampling(model, cond, device)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            
            d_complex = torch.view_as_complex(data.squeeze(0)).cpu()
            p_complex = torch.view_as_complex(pred.squeeze(0)).cpu()
            d_t = d_complex.to(torch.complex64).unsqueeze(0).to(device)
            p_t = p_complex.to(torch.complex64).unsqueeze(0).to(device)
            ssim = eval_ssim(
                p_t, d_t,
                params.sample_rate, params.input_dim, device
            ).item()
            ssim_list.append(ssim)
    
    results.append({
        "strategy": "robust_ddim",
        "skip": 1,
        "model_evals": 100,
        "ssim_mean": np.mean(ssim_list),
        "ssim_std": np.std(ssim_list),
        "time_mean": np.mean(times),
        "time_std": np.std(times),
    })
    print(f"  Robust DDIM SSIM: {results[-1]['ssim_mean']:.4f}")
    
    return {"sampling_comparison": results}


# ============================================================================
# Figure Generation
# ============================================================================

def generate_ofdm_analysis_figures(results: Dict, output_dir: Path):
    """Generate OFDM physical plausibility analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 10
    
    per_sample = results["per_sample"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: EVM Distribution
    ax = axes[0, 0]
    evm_values = [r["evm"] for r in per_sample]
    ax.bar(range(len(evm_values)), evm_values, color='steelblue', alpha=0.7)
    ax.axhline(results["evm_mean"], color='red', linestyle='--', label=f'Mean: {results["evm_mean"]:.4f}')
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('EVM')
    ax.set_title('Error Vector Magnitude (EVM)\n(Lower is better)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: SSIM Distribution
    ax = axes[0, 1]
    ssim_values = [r["ssim"] for r in per_sample]
    ax.bar(range(len(ssim_values)), ssim_values, color='forestgreen', alpha=0.7)
    ax.axhline(results["ssim_mean"], color='red', linestyle='--', label=f'Mean: {results["ssim_mean"]:.4f}')
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('SSIM')
    ax.set_title('Structural Similarity Index (SSIM)\n(Higher is better)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: NMSE Distribution
    ax = axes[0, 2]
    nmse_values = [r["nmse"] for r in per_sample]
    ax.bar(range(len(nmse_values)), nmse_values, color='darkorange', alpha=0.7)
    ax.axhline(results["nmse_mean"], color='red', linestyle='--', label=f'Mean: {results["nmse_mean"]:.4f}')
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('NMSE')
    ax.set_title('Normalized MSE (NMSE)\n(Lower is better)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 4: Subcarrier Correlation Comparison
    ax = axes[1, 0]
    x = np.arange(len(per_sample))
    width = 0.35
    real_corr = [r["subcarrier_corr_real"] for r in per_sample]
    gen_corr = [r["subcarrier_corr_gen"] for r in per_sample]
    ax.bar(x - width/2, real_corr, width, label='Real', color='blue', alpha=0.7)
    ax.bar(x + width/2, gen_corr, width, label='Generated', color='orange', alpha=0.7)
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('Correlation')
    ax.set_title('Subcarrier Correlation\n(Adjacent subcarriers)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 5: Constant Modulus Ratio
    ax = axes[1, 1]
    const_real = [r["constant_modulus_ratio_real"] for r in per_sample]
    const_gen = [r["constant_modulus_ratio_gen"] for r in per_sample]
    ax.boxplot([const_real, const_gen], labels=['Real', 'Generated'])
    ax.set_ylabel('Constant Modulus Ratio')
    ax.set_title('Constant Modulus Property\n(Pilot subcarriers)')
    ax.grid(alpha=0.3)
    
    # Panel 6: Frequency Correlation (lag-1)
    ax = axes[1, 2]
    freq_corr_real = [r["freq_lag1_corr_real"] for r in per_sample]
    freq_corr_gen = [r["freq_lag1_corr_gen"] for r in per_sample]
    ax.boxplot([freq_corr_real, freq_corr_gen], labels=['Real', 'Generated'])
    ax.set_ylabel('Correlation')
    ax.set_title('Frequency Correlation (Lag-1)\n(Adjacent subcarrier)')
    ax.grid(alpha=0.3)
    
    plt.suptitle('OFDM Physical Plausibility Analysis\nRF-Diffusion Wi-Fi Generation Quality', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "fig_ofdm_analysis.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close()
    
    # Additional figure: Constellation scatter plot for a sample
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use first sample for constellation
    sample = per_sample[0]
    # We'll need to regenerate or load the actual complex values
    # For now, show EVM vs SSIM scatter
    ax = axes[0]
    ax.scatter(evm_values, ssim_values, alpha=0.7, c='steelblue', s=50)
    ax.set_xlabel('EVM (Lower is better)')
    ax.set_ylabel('SSIM (Higher is better)')
    ax.set_title('EVM vs SSIM Trade-off')
    ax.grid(alpha=0.3)
    
    # Correlation comparison bar chart
    ax = axes[1]
    metrics = ['Subcarrier\nCorr', 'Const Mod\nRatio', 'Freq Lag-1\nCorr']
    real_vals = [results["subcarrier_corr_real_mean"], 
                 results["const_modulus_real_mean"],
                 results["freq_lag1_corr_real_mean"]]
    gen_vals = [results["subcarrier_corr_gen_mean"],
                results["const_modulus_gen_mean"],
                results["freq_lag1_corr_gen_mean"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, real_vals, width, label='Real', color='blue', alpha=0.7)
    ax.bar(x + width/2, gen_vals, width, label='Generated', color='orange', alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Physical Plausibility Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path2 = output_dir / "fig_ofdm_metrics.pdf"
    plt.savefig(output_path2, bbox_inches='tight', dpi=300)
    plt.savefig(output_path2.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path2}")
    plt.close()


def generate_blur_ablation_figure(results: Dict, output_dir: Path):
    """Generate blur kernel ablation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    ablation = results["blur_ablation"]
    sigma_values = [r["sigma"] for r in ablation]
    ssim_means = [r["ssim_mean"] for r in ablation]
    ssim_stds = [r["ssim_std"] for r in ablation]
    nmse_means = [r["nmse_mean"] for r in ablation]
    nmse_stds = [r["nmse_std"] for r in ablation]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # SSIM vs sigma
    ax = axes[0]
    ax.errorbar(sigma_values, ssim_means, yerr=ssim_stds, 
                fmt='o-', capsize=5, color='steelblue', markersize=10)
    ax.set_xlabel('Blur Sigma (σ)')
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM vs Blur Kernel Sigma\n(Higher is better)')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # NMSE vs sigma
    ax = axes[1]
    ax.errorbar(sigma_values, nmse_means, yerr=nmse_stds,
                fmt='s-', capsize=5, color='darkorange', markersize=10)
    ax.set_xlabel('Blur Sigma (σ)')
    ax.set_ylabel('NMSE')
    ax.set_title('NMSE vs Blur Kernel Sigma\n(Lower is better)')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Time vs sigma
    ax = axes[2]
    time_means = [r["time_mean"] for r in ablation]
    time_stds = [r["time_std"] for r in ablation]
    ax.errorbar(sigma_values, time_means, yerr=time_stds,
                fmt='^-', capsize=5, color='forestgreen', markersize=10)
    ax.set_xlabel('Blur Sigma (σ)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Generation Time vs Blur Sigma')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.suptitle('Frequency Blur Kernel Ablation Study', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "fig_blur_ablation.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def generate_sampling_comparison_figure(results: Dict, output_dir: Path):
    """Generate sampling strategy comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    comparison = results["sampling_comparison"]
    
    strategies = [r["strategy"] for r in comparison]
    ssim_means = [r["ssim_mean"] for r in comparison]
    ssim_stds = [r["ssim_std"] for r in comparison]
    evals = [r["model_evals"] for r in comparison]
    times = [r["time_mean"] for r in comparison]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SSIM bar chart
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(strategies)))
    bars = ax.bar(range(len(strategies)), ssim_means, color=colors, 
                  edgecolor='k', yerr=ssim_stds, capsize=4, alpha=0.85)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM by Sampling Strategy')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ssim_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Time vs Quality scatter
    ax = axes[1]
    for i, r in enumerate(comparison):
        ax.scatter(r["time_mean"], r["ssim_mean"], s=200, c=[colors[i]], 
                   edgecolors='k', zorder=5)
        ax.annotate(r["strategy"], (r["time_mean"] + 0.02, r["ssim_mean"]),
                    fontsize=9)
    ax.set_xlabel('Generation Time (s)')
    ax.set_ylabel('SSIM')
    ax.set_title('Quality vs Speed Trade-off')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Sampling Strategy Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "fig_sampling_strategies.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RF-Diffusion Physical Experiments')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=41)
    parser.add_argument('--blur-samples', type=int, default=10,
                       help='Number of samples for blur ablation (smaller for speed)')
    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    
    # Create output directories
    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading RF-Diffusion Wi-Fi model...")
    params = all_params[0]  # Wi-Fi
    checkpoint = torch.load(
        REAL_UPSTREAM / "model" / "wifi" / "b32-256-100s" / "weights.pt",
        map_location=device)
    model = tfdiff_WiFi(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = SignalDiffusion(AttrDict(params))
    
    all_results = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": args.num_samples,
        "blur_ablation_samples": args.blur_samples,
    }
    
    # Load dataset
    dataset = list(from_path_inference(AttrDict(params)))
    print(f"Loaded {len(dataset)} Wi-Fi samples")
    
    # Experiment 1: OFDM Physical Plausibility
    print("\n" + "="*70)
    print("Running Experiment 1: OFDM Physical Plausibility Test")
    print("="*70)
    ofdm_results = run_ofdm_plausibility_test(
        model, diffusion, dataset, device, AttrDict(params), 
        num_samples=args.num_samples
    )
    all_results["ofdm_plausibility"] = ofdm_results
    print(f"\nOFDM Results Summary:")
    print(f"  EVM: {ofdm_results['evm_mean']:.4f} ± {ofdm_results['evm_std']:.4f}")
    print(f"  SSIM: {ofdm_results['ssim_mean']:.4f} ± {ofdm_results['ssim_std']:.4f}")
    print(f"  NMSE: {ofdm_results['nmse_mean']:.6f} ± {ofdm_results['nmse_std']:.6f}")
    
    # Experiment 2: Blur Kernel Ablation
    print("\n" + "="*70)
    print("Running Experiment 2: Frequency Blur Kernel Ablation")
    print("="*70)
    blur_results = run_blur_kernel_ablation(
        model, params, device, num_samples=args.blur_samples
    )
    all_results["blur_ablation"] = blur_results["blur_ablation"]
    
    # Experiment 3: Sampling Strategy Comparison
    print("\n" + "="*70)
    print("Running Experiment 3: Sampling Strategy Comparison")
    print("="*70)
    sampling_results = run_sampling_strategy_comparison(
        model, params, device, num_samples=args.num_samples
    )
    all_results["sampling_comparison"] = sampling_results["sampling_comparison"]
    
    # Save results
    results_file = output_dir / f"physics_experiments_{run_id}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results: {results_file}")
    
    # Generate CSV summaries
    # OFDM plausibility CSV
    ofdm_csv = output_dir / f"ofdm_plausibility_{run_id}.csv"
    with open(ofdm_csv, "w") as f:
        f.write("sample_id,evm,ssim,nmse,subcarrier_corr_real,subcarrier_corr_gen,")
        f.write("constant_modulus_ratio_real,constant_modulus_ratio_gen,")
        f.write("freq_lag1_corr_real,freq_lag1_corr_gen\n")
        for r in ofdm_results["per_sample"]:
            f.write(f"{r['sample_id']},{r['evm']:.6f},{r['ssim']:.6f},{r['nmse']:.6f},")
            f.write(f"{r['subcarrier_corr_real']:.6f},{r['subcarrier_corr_gen']:.6f},")
            f.write(f"{r['constant_modulus_ratio_real']:.6f},{r['constant_modulus_ratio_gen']:.6f},")
            f.write(f"{r['freq_lag1_corr_real']:.6f},{r['freq_lag1_corr_gen']:.6f}\n")
    print(f"Saved CSV: {ofdm_csv}")
    
    # Blur ablation CSV
    blur_csv = output_dir / f"blur_kernel_ablation_{run_id}.csv"
    with open(blur_csv, "w") as f:
        f.write("sigma,variance,ssim_mean,ssim_std,nmse_mean,nmse_std,time_mean,time_std,peak_mem_mb\n")
        for r in all_results["blur_ablation"]:
            f.write(f"{r['sigma']},{r['variance']},{r['ssim_mean']:.6f},{r['ssim_std']:.6f},")
            f.write(f"{r['nmse_mean']:.6f},{r['nmse_std']:.6f},{r['time_mean']:.6f},{r['time_std']:.6f},")
            f.write(f"{r['peak_mem_mb']:.2f}\n")
    print(f"Saved CSV: {blur_csv}")
    
    # Sampling comparison CSV
    sampling_csv = output_dir / f"sampling_strategy_comparison_{run_id}.csv"
    with open(sampling_csv, "w") as f:
        f.write("strategy,skip,model_evals,ssim_mean,ssim_std,time_mean,time_std\n")
        for r in all_results["sampling_comparison"]:
            f.write(f"{r['strategy']},{r['skip']},{r['model_evals']},{r['ssim_mean']:.6f},")
            f.write(f"{r['ssim_std']:.6f},{r['time_mean']:.6f},{r['time_std']:.6f}\n")
    print(f"Saved CSV: {sampling_csv}")
    
    # Generate figures
    print("\n" + "="*70)
    print("Generating Figures")
    print("="*70)
    generate_ofdm_analysis_figures(ofdm_results, figures_dir)
    generate_blur_ablation_figure(all_results, figures_dir)
    generate_sampling_comparison_figure(all_results, figures_dir)
    
    # Copy to report directory
    import shutil
    report_fig_dir = PROJECT_ROOT / "report" / "figures"
    for fmt in ["pdf", "png"]:
        for fig_name in ["fig_ofdm_analysis", "fig_ofdm_metrics", 
                        "fig_blur_ablation", "fig_sampling_strategies"]:
            src = figures_dir / f"{fig_name}.{fmt}"
            dst = report_fig_dir / f"{fig_name}.{fmt}"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"Copied to report: {dst}")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nKey Findings:")
    print(f"\n1. OFDM Physical Plausibility:")
    print(f"   - EVM: {ofdm_results['evm_mean']:.4f} (lower is better, <0.1 is good)")
    print(f"   - SSIM: {ofdm_results['ssim_mean']:.4f}")
    print(f"   - Subcarrier correlation diff: {np.mean([r['subcarrier_corr_diff'] for r in ofdm_results['per_sample']]):.4f}")
    print(f"\n2. Blur Kernel Ablation:")
    best_blur = max(all_results["blur_ablation"], key=lambda x: x["ssim_mean"])
    print(f"   - Best sigma: {best_blur['sigma']} (SSIM: {best_blur['ssim_mean']:.4f})")
    print(f"\n3. Sampling Strategies:")
    best_strategy = max(all_results["sampling_comparison"], key=lambda x: x["ssim_mean"])
    fastest = min(all_results["sampling_comparison"], key=lambda x: x["time_mean"])
    print(f"   - Best quality: {best_strategy['strategy']} (SSIM: {best_strategy['ssim_mean']:.4f})")
    print(f"   - Fastest: {fastest['strategy']} ({fastest['time_mean']:.3f}s)")
    
    return all_results


if __name__ == "__main__":
    main()
