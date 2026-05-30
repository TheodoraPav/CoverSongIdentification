"""Augmentation profiles: `time` (waveform).

Public API:
    build_time_augmenter()                  -> audiomentations.Compose
    apply_waveform_augment(aug, x, sr, seed)
    length_fix(samples, target_length)
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



