# Evaluation Methodology

This document describes the metrics and procedures used to assess the quality of RF-Diffusion's generated radio signals.

## Metrics

### SSIM (Structural Similarity Index)

SSIM measures the structural similarity between generated and ground-truth signals in the time domain. Unlike pixel-wise metrics such as MSE, SSIM accounts for structural information:

$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

The terms are defined as follows: $\mu_x, \mu_y$ are the mean values of each signal, $\sigma_x^2, \sigma_y^2$ are the variances, $\sigma_{xy}$ is the covariance, and $C_1, C_2$ are small stabilization constants. SSIM ranges from 0 to 1, where 1 indicates perfect structural similarity.

### FID (Fréchet Inception Distance)

FID computes the Wasserstein-2 distance between feature distributions extracted from a pre-trained Inception network applied to time-frequency spectrograms:

$$\text{FID} = \|\mu_1 - \mu_2\|^2 + \text{Tr}(\Sigma_1 + \Sigma_2 - 2\sqrt{\Sigma_1 \Sigma_2})$$

Here $\mu_1, \mu_2$ are the feature means, $\Sigma_1, \Sigma_2$ are the covariance matrices, and $\text{Tr}$ denotes the trace. Lower FID indicates better quality; FID = 0 means the two distributions are identical.

### SNR (Signal-to-Noise Ratio)

For 5G MIMO channel estimation, SNR is defined as:

$$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$

where noise is the difference between the predicted channel and the ground truth. Higher SNR corresponds to better channel estimation accuracy.

## Evaluation Protocol

### Wi-Fi Generation

A pre-trained Wi-Fi model (32-block, 256 hidden dimensions) is used to generate 215 samples under the official test conditions. SSIM is computed between generated and ground-truth time-domain signals. FID is computed between spectrogram distributions.

### 5G MIMO Channel Estimation

A pre-trained MIMO model generates channel estimates using fast sampling. SNR is computed between estimated and ground-truth channels.

### Efficiency Analysis

Experiments sweep model configurations (8, 16, 32 blocks) and sampling step counts (10, 20, 50, 100). Three sampling strategies are compared: native (DDPM), fast, and full reverse (DDIM). For each configuration, SSIM, inference time, and peak GPU memory are recorded.

## Reproducibility

All experiments use a fixed random seed of 11. PyTorch version is 2.0 or higher, with minor variations in SSIM below 0.01. CUDA versions 11.8+ and 12.x are both supported. The primary evaluation script is `scripts/run_wifi_sampling.py`.

## Results Summary

| Metric | Wi-Fi | 5G MIMO |
|--------|-------|---------|
| Average SSIM | 0.81 | N/A (channel-estimation metric) |
| FID | 7.82 | N/A (out of scope) |
| Average SNR (dB) | N/A (CSI tasks use SSIM/FID) | 29.95 |

For 5G MIMO channel estimation, the standard ranking metric is SNR (dB); SSIM and FID were not reported in the upstream paper for this task.

## Limitations

SSIM may not fully capture perceptual quality aspects of RF signals. FID depends on the quality of the feature extractor; for RF signals in particular, the standard Inception-based approach may not be optimal. SNR assumes an additive Gaussian noise model, which may not hold in all propagation scenarios.
