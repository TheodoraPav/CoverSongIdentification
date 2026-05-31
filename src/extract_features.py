"""Extract pooled backbone embeddings and cache them to `features.pt`.

Pipeline per segment:
    load wav clip -> resample -> (optional) augmentation -> length fix
    -> backbone (frozen) -> temporal pooling -> L2 normalize

Usage:
    python src/extract_features.py --config configs/<experiment>.yaml
    python src/extract_features.py --config configs/<experiment>.yaml --batch-size 4

This module also exposes helpers used by `train.py` and `evaluate.py`:
    forward_batch, pooled_features_from_batch.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentations import (  # noqa: E402
    apply_waveform_augment,
    build_time_augmenter,
    length_fix,
)
from src.model import (  # noqa: E402
    backbone_forward,
    get_backbone_spec,
    load_backbone,
    pool_backbone_output,
)
from src.utils import (  # noqa: E402
    ExperimentConfig,
    features_file_for,
    get_logger,
    load_config,
    pick_device,
    resolve_audio_path,
    segment_seed,
    segments_file_for,
    set_global_seed,
)

LOGGER = get_logger("extract_features")


# -----------------------------------------------------------------------------
# Segments table
# -----------------------------------------------------------------------------


SEGMENT_COLUMNS = (
    "track_id",
    "group_id",
    "role",
    "audio_path",
    "seg_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "sampling",
)


def load_segments_table(cfg: ExperimentConfig) -> pd.DataFrame:
    path = segments_file_for(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Segments file not found: {path}. "
            "Run preprocess_segments.py first."
        )

    df = pd.read_csv(path)
    missing = set(SEGMENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    subset = df[df["sampling"] == cfg.sampling].reset_index(drop=True)
    if len(subset) == 0:
        raise ValueError(
            f"No rows with sampling={cfg.sampling!r} in {path.name}. "
            f"Available: {sorted(df['sampling'].unique())}"
        )
    if len(subset) < len(df):
        LOGGER.info("Using %d / %d rows (sampling=%s).", len(subset), len(df), cfg.sampling)
    return subset


# -----------------------------------------------------------------------------
# Audio loading + per-segment pre-processing
# -----------------------------------------------------------------------------


def load_segment_waveform(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    target_sr: int,
) -> np.ndarray:
    """Load one mono float32 clip and resample to `target_sr`.

    Uses librosa (not ``torchaudio.info`` / partial load) for compatibility with
    torchaudio 2.9+ on Colab/Kaggle.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    duration = max(1e-3, float(end_sec) - float(start_sec))
    wav, _sr = librosa.load(
        str(audio_path),
        sr=target_sr,
        mono=True,
        offset=float(start_sec),
        duration=duration,
    )
    return wav.astype(np.float32)


def prepare_waveform(
    samples: np.ndarray,
    cfg: ExperimentConfig,
    sample_rate: int,
    group_id: int,
    seg_id: int,
    time_augmenter,
    apply_aug: bool,
) -> np.ndarray:
    """Apply optional waveform augmentation and pad/truncate to fixed length."""
    out = samples
    if apply_aug and cfg.augment == "time":
        seed = segment_seed(cfg.seed, group_id, seg_id)
        out = apply_waveform_augment(time_augmenter, out, sample_rate, seed)
    if cfg.sampling != "mixed":
        target_len = int(round(cfg.segment_seconds * sample_rate))
        out = length_fix(out, target_len)
    return out.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Backbone forward for a batch
# -----------------------------------------------------------------------------


def forward_batch(
    model: torch.nn.Module,
    processor: object,
    spec,
    waveforms: list[np.ndarray],
    pool: str,
    device: torch.device,
) -> torch.Tensor:
    """Run frozen backbone + pooling on a list of 1D waveforms."""
    hidden, mask = backbone_forward(
        model,
        processor,
        waveforms,
        spec.sample_rate,
        spec,
        device=device,
        mel_transform=None,
    )
    return pool_backbone_output(hidden, mask, method=pool)


# -----------------------------------------------------------------------------
# Per-batch helpers shared with train.py and evaluate.py
# -----------------------------------------------------------------------------
def pooled_features_from_batch(
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """Return pooled L2-normalized backbone features for a DataLoader batch."""
    return batch["features"].to(device)


# -----------------------------------------------------------------------------
# Extraction loop
# -----------------------------------------------------------------------------


def should_apply_offline_augment(cfg: ExperimentConfig) -> bool:
    if cfg.augment == "none":
        return False
    return True


def _load_batch(
    cfg: ExperimentConfig,
    rows: pd.DataFrame,
    spec,
    time_augmenter,
    apply_aug: bool,
    progress: tqdm,
) -> tuple[list[np.ndarray], list[int], list[pd.Series]]:
    """Load + augment all rows in a chunk; missing audio is skipped."""
    waveforms: list[np.ndarray] = []
    seeds: list[int] = []
    ok_rows: list[pd.Series] = []

    for _, row in rows.iterrows():
        audio_path = resolve_audio_path(cfg, str(row["audio_path"]))
        try:
            wav = load_segment_waveform(
                audio_path,
                float(row["start_sec"]),
                float(row["end_sec"]),
                spec.sample_rate,
            )
        except FileNotFoundError:
            LOGGER.warning("Skipping missing audio: %s", audio_path)
            progress.update(1)
            continue

        gid = int(row["group_id"])
        sid = int(row["seg_id"])
        wav = prepare_waveform(wav, cfg, spec.sample_rate, gid, sid, time_augmenter, apply_aug)

        waveforms.append(wav)
        seeds.append(segment_seed(cfg.seed, gid, sid))
        ok_rows.append(row)

    return waveforms, seeds, ok_rows


def _validate_segment_audio_paths(
    cfg: ExperimentConfig, segments: pd.DataFrame, n_check: int = 5
) -> None:
    """Fail fast before loading the backbone if paths are wrong."""
    n = min(n_check, len(segments))
    if n == 0:
        raise RuntimeError("segments table is empty.")
    missing: list[Path] = []
    for i in range(n):
        p = resolve_audio_path(cfg, str(segments.iloc[i]["audio_path"]))
        if not p.is_file():
            missing.append(p)
    if len(missing) == n:
        sample = missing[0]
        raise FileNotFoundError(
            f"No audio found for the first {n} segments (example: {sample}). "
            "Check paths.manifest and paths.audio_root in the YAML."
        )


def extract_all(cfg: ExperimentConfig, batch_size: int = 8) -> dict:
    set_global_seed(cfg.seed)
    device = pick_device()

    segments = load_segments_table(cfg)
    _validate_segment_audio_paths(cfg, segments)
    apply_aug = should_apply_offline_augment(cfg)
    augment_flag = cfg.augment if apply_aug else "none"

    time_augmenter = None
    if apply_aug and cfg.augment == "time":
        time_augmenter = build_time_augmenter()

    model, processor, spec = load_backbone(
        cfg.backbone,
        checkpoint=cfg.backbone_checkpoint,
        device=device,
    )

    features: list[torch.Tensor] = []
    metadata: dict[str, list] = {
        "track_id": [],
        "group_id": [],
        "role": [],
        "seg_id": [],
    }

    n = len(segments)
    progress = tqdm(total=n, desc="extract", unit="seg")

    for start in range(0, n, batch_size):
        rows = segments.iloc[start : start + batch_size]
        waveforms, seeds, ok_rows = _load_batch(
            cfg, rows, spec, time_augmenter, apply_aug, progress
        )
        if not waveforms:
            continue

        pooled = forward_batch(
            model, processor, spec, waveforms, cfg.pool, device
        )

        for i, row in enumerate(ok_rows):
            features.append(pooled[i].cpu())
            metadata["track_id"].append(str(row["track_id"]))
            metadata["group_id"].append(int(row["group_id"]))
            metadata["role"].append(str(row["role"]))
            metadata["seg_id"].append(int(row["seg_id"]))

        progress.update(len(ok_rows))

    progress.close()

    if not features:
        raise RuntimeError("No features extracted. Check audio paths and segments.")

    feature_tensor = torch.stack(features, dim=0)
    hidden_dim = int(feature_tensor.shape[1])

    LOGGER.info(
        "Extracted %d vectors (dim=%d, backbone=%s, augment=%s, sampling=%s).",
        feature_tensor.shape[0],
        hidden_dim,
        cfg.backbone,
        augment_flag,
        cfg.sampling,
    )

    # Free backbone from GPU before returning cached features
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "features": feature_tensor,
        **metadata,
        "hidden_dim": hidden_dim,
        "backbone": cfg.backbone,
        "backbone_checkpoint": cfg.backbone_checkpoint,
        "augment": augment_flag,
        "sampling": cfg.sampling,
        "pool": cfg.pool,
        "segment_pool_mode": cfg.segment_pool_mode,
        "segment_pool_max": cfg.segment_pool_max,
        "segments_per_track": cfg.segments_per_track,
        "experiment_name": cfg.experiment_name,
    }


def save_features(payload: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)
    LOGGER.info("Wrote %s (%d segments).", dest, payload["features"].shape[0])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_cli() -> tuple[ExperimentConfig, int]:
    parser = argparse.ArgumentParser(
        description="Extract and cache pooled backbone embeddings."
    )
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Segments per GPU forward pass (default: 8).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    return load_config(args.config), int(args.batch_size)


def main() -> None:
    cfg, batch_size = parse_cli()
    payload = extract_all(cfg, batch_size=batch_size)
    save_features(payload, features_file_for(cfg))


if __name__ == "__main__":
    main()
