# Algorithm Overview

## RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion

RF-Diffusion is a diffusion-based generative model for radio frequency (RF) signal generation, introduced in [MobiCom 2024](https://doi.org/10.1145/3636534.3649348).

## Core Concept

RF-Diffusion generates radio signals through three key design choices. First, RF signals are converted to a time-frequency representation using Short-Time Fourier Transform (STFT), a transformation that makes the underlying signal structure more amenable to diffusion modeling. Second, the model operates directly on complex-valued spectrograms, preserving both amplitude and phase information that are essential for RF applications. Third, a multi-block transformer architecture with skip connections enables efficient gradient flow during the reverse diffusion process.

## Forward Diffusion Process

Given a clean signal $x_0$, the forward diffusion process progressively adds Gaussian noise over $T$ timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right)$$

where $\beta_t$ is a noise schedule. After $T$ steps, $x_T$ approximates isotropic Gaussian noise.

## Reverse Diffusion Process

The reverse process learns to denoise, parameterized by a neural network $\epsilon_\theta$:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \sigma_t I\right)$$

## Model Architecture

The backbone is a complex-valued U-Net or Transformer that takes as input the time-frequency representation of RF signals (real and imaginary channels), processes them through stacked transformer blocks with self-attention and cross-attention, and outputs a complex-valued spectrogram matching the input dimensions.

## Conditioning

RF-Diffusion supports conditional generation, where auxiliary information such as CSI estimates is provided as input:

$$p_\theta(x | c) = \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t, c)$$

where $c$ represents the conditioning signal.

## Supported Tasks

- **Wi-Fi Sensing**: Generate Wi-Fi Channel State Information (CSI) for sensing applications
- **FMCW Radar**: Generate Frequency Modulated Continuous Wave radar signals
- **5G MIMO Channel Estimation**: Estimate MIMO channel coefficients from limited observations

## Sampling Strategies

Three sampling strategies are available. Native sampling (DDPM) uses the standard reverse diffusion process with full timesteps. Fast sampling leverages learned skip connections for single-step generation. DDIM uses fewer steps with a deterministic trajectory.

## Key Innovations

The model contributes three main advances. Time-frequency domain diffusion addresses the challenge of modeling non-stationary RF signals. Complex-valued architecture preserves phase information that is critical for RF applications. Hierarchical generation provides a multi-resolution approach for efficient computation.
