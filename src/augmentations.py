"""Augmentation profiles: `time` (waveform) and `spec` (SpecAugment).

Public API:
    build_time_augmenter()                  -> audiomentations.Compose
    apply_waveform_augment(aug, x, sr, seed)
    length_fix(samples, target_length)
    apply_spec_augment(mel, seed, ...)
"""

from __future__ import annotations

import numpy as np
import torch


# -----------------------------------------------------------------------------
# `time` profile: waveform augmentation via audiomentations
# -----------------------------------------------------------------------------


def build_time_augmenter():
    """
    Each transform applies independently with its listed probability.
    Imports `audiomentations` lazily so code paths that only use SpecAugment
    don't pay the import cost.
    """
    from audiomentations import (
        AddGaussianSNR,
        Compose,
        Gain,
        HighShelfFilter,
        LowShelfFilter,
        PitchShift,
        TimeStretch,
    )

    return Compose(
        [
            PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.6),
            TimeStretch(
                min_rate=0.92,
                max_rate=1.08,
                leave_length_unchanged=True,
                p=0.4,
            ),
            Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.5),
            AddGaussianSNR(min_snr_db=25.0, max_snr_db=40.0, p=0.3),
            LowShelfFilter(min_gain_db=-4.0, max_gain_db=4.0, p=0.25),
            HighShelfFilter(min_gain_db=-4.0, max_gain_db=4.0, p=0.25),
        ]
    )


def apply_waveform_augment(
    augmenter,
    samples: np.ndarray,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    arr = samples.astype(np.float32, copy=False)
    state = np.random.get_state()
    np.random.seed(int(seed) & 0x7FFFFFFF)
    try:
        return augmenter(samples=arr, sample_rate=sample_rate)
    finally:
        np.random.set_state(state)


def length_fix(samples: np.ndarray, target_length: int) -> np.ndarray:
    n = samples.shape[-1]
    if n == target_length:
        return samples
    if n > target_length:
        return samples[..., :target_length]
    pad_width = [(0, 0)] * (samples.ndim - 1) + [(0, target_length - n)]
    return np.pad(samples, pad_width, mode="constant")


# -----------------------------------------------------------------------------
# `spec` profile: SpecAugment for AST log-mel input
# -----------------------------------------------------------------------------


def apply_spec_augment(
    mel: torch.Tensor,
    seed: int,
    n_freq_masks: int = 2,
    max_freq_width: int = 48,
    n_time_masks: int = 2,
    max_time_width: int = 96,
) -> torch.Tensor:
    """SpecAugment on a 2D log-mel `(n_mels, T)` (AST backbone only).

    Up to `n_freq_masks` horizontal bands (width <= `max_freq_width`) and
    `n_time_masks` vertical bands (width <= `max_time_width`) are zeroed.
    """
    if mel.dim() != 2:
        raise ValueError(f"Expected 2D (n_mels, T), got {tuple(mel.shape)}")

    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    out = mel.clone()
    n_mels, n_time = out.shape

    k_f = int(torch.randint(1, n_freq_masks + 1, (1,), generator=g).item())
    for _ in range(k_f):
        w = int(torch.randint(0, max_freq_width + 1, (1,), generator=g).item())
        if w == 0 or w >= n_mels:
            continue
        f0 = int(torch.randint(0, n_mels - w, (1,), generator=g).item())
        out[f0 : f0 + w, :] = 0.0

    k_t = int(torch.randint(1, n_time_masks + 1, (1,), generator=g).item())
    for _ in range(k_t):
        w = int(torch.randint(0, max_time_width + 1, (1,), generator=g).item())
        if w == 0 or w >= n_time:
            continue
        t0 = int(torch.randint(0, n_time - w, (1,), generator=g).item())
        out[:, t0 : t0 + w] = 0.0

    return out
