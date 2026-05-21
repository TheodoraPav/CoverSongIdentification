"""PyTorch datasets and DataLoaders for training and evaluation.

`augment_mode: offline` (default)
    Loads pooled vectors from `features.pt` produced by `extract_features.py`.

`augment_mode: online`
    Loads wav segments from `segments.csv` and applies augmentation each epoch
    (fresh randomness via `set_epoch`).

Public API:
    split_group_ids(cfg, all_group_ids) -> (train_groups, val_groups)
    CachedFeatureDataset
    OnlineSegmentDataset
    build_dataloaders(cfg) -> (train_loader, val_loader)
    set_epoch(loader, epoch)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentations import (  # noqa: E402
    apply_waveform_augment,
    build_time_augmenter,
    length_fix,
)
from src.extract_features import (  # noqa: E402
    load_segment_waveform,
    load_segments_table,
)
from src.model import get_backbone_spec  # noqa: E402
from src.utils import (  # noqa: E402
    ExperimentConfig,
    features_file_for,
    resolve_audio_path,
    segment_seed,
)

# -----------------------------------------------------------------------------
# Train / validation split (by group_id, no leakage)
# -----------------------------------------------------------------------------


def split_group_ids(
    cfg: ExperimentConfig,
    all_group_ids: list[int] | np.ndarray,
) -> tuple[set[int], set[int]]:
    """Split cover-song groups into train and validation sets."""
    unique = sorted({int(g) for g in all_group_ids})
    if len(unique) < 2:
        raise ValueError(
            f"Need at least 2 group_id values for a train/val split, got {len(unique)}"
        )

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(unique)

    n_val = max(1, int(round(len(unique) * cfg.training.val_fraction)))
    n_val = min(n_val, len(unique) - 1)
    val_groups = set(unique[:n_val])
    train_groups = set(unique[n_val:])
    return train_groups, val_groups


def _indices_for_groups(group_ids: list[int], allowed: set[int]) -> list[int]:
    return [i for i, g in enumerate(group_ids) if int(g) in allowed]


# -----------------------------------------------------------------------------
# Feature cache (offline mode)
# -----------------------------------------------------------------------------


def load_feature_cache(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {path}. "
            "Run extract_features.py first (augment_mode: offline)."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError(f"Invalid cache format in {path}")
    return payload


def _check_cache_matches_cfg(payload: dict, cfg: ExperimentConfig) -> None:
    for key, expected in (("backbone", cfg.backbone), ("sampling", cfg.sampling)):
        found = payload.get(key)
        if found is not None and found != expected:
            raise ValueError(
                f"Cache {key}={found!r} does not match config {expected!r}. "
                "Re-run extract_features.py with this config."
            )

    if cfg.augment_mode == "offline" and cfg.augment != "none":
        cached_aug = payload.get("augment", "none")
        if cached_aug != cfg.augment:
            raise ValueError(
                f"Cache augment={cached_aug!r} does not match config "
                f"augment={cfg.augment!r}. Re-run extract_features.py."
            )


class CachedFeatureDataset(Dataset):
    """One row per segment: pooled backbone vector + metadata."""

    def __init__(
        self,
        features: torch.Tensor,
        group_ids: list[int],
        roles: list[str],
        track_ids: list[str],
        seg_ids: list[int],
        indices: list[int],
    ) -> None:
        self.features = features
        self.group_ids = group_ids
        self.roles = roles
        self.track_ids = track_ids
        self.seg_ids = seg_ids
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        return {
            "feature": self.features[idx],
            "group_id": int(self.group_ids[idx]),
            "role": str(self.roles[idx]),
            "track_id": str(self.track_ids[idx]),
            "seg_id": int(self.seg_ids[idx]),
        }


# -----------------------------------------------------------------------------
# Raw segments (online mode)
# -----------------------------------------------------------------------------


class OnlineSegmentDataset(Dataset):
    """Load wav clips; augmentation changes each epoch via `set_epoch`."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        segments: pd.DataFrame,
        indices: list[int],
    ) -> None:
        self.cfg = cfg
        self.segments = segments
        self.indices = indices
        self.epoch = 0
        self.backbone_spec = get_backbone_spec(cfg.backbone)
        self.time_augmenter = build_time_augmenter() if cfg.augment == "time" else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        row = self.segments.iloc[self.indices[i]]
        audio_path = resolve_audio_path(self.cfg, str(row["audio_path"]))
        sr = self.backbone_spec.sample_rate

        wav = load_segment_waveform(
            audio_path,
            float(row["start_sec"]),
            float(row["end_sec"]),
            sr,
        )

        gid = int(row["group_id"])
        sid = int(row["seg_id"])

        if self.time_augmenter is not None:
            seed = segment_seed(self.cfg.seed + self.epoch * 1_000_003, gid, sid)
            wav = apply_waveform_augment(self.time_augmenter, wav, sr, seed)

        if self.cfg.sampling != "mixed":
            target_len = int(round(self.cfg.segment_seconds * sr))
            wav = length_fix(wav, target_len)

        return {
            "waveform": torch.from_numpy(wav),
            "group_id": gid,
            "role": str(row["role"]),
            "track_id": str(row["track_id"]),
            "seg_id": sid,
        }


# -----------------------------------------------------------------------------
# Collate
# -----------------------------------------------------------------------------


def collate_cached(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([b["feature"] for b in batch], dim=0),
        "group_id": torch.tensor([b["group_id"] for b in batch], dtype=torch.long),
        "role": [b["role"] for b in batch],
        "track_id": [b["track_id"] for b in batch],
        "seg_id": torch.tensor([b["seg_id"] for b in batch], dtype=torch.long),
    }


def collate_online(batch: list[dict]) -> dict:
    waveforms = [b["waveform"] for b in batch]
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    lengths = torch.zeros(len(waveforms), dtype=torch.long)
    for i, w in enumerate(waveforms):
        n = w.shape[0]
        padded[i, :n] = w
        lengths[i] = n

    return {
        "waveforms": padded,
        "lengths": lengths,
        "group_id": torch.tensor([b["group_id"] for b in batch], dtype=torch.long),
        "role": [b["role"] for b in batch],
        "track_id": [b["track_id"] for b in batch],
        "seg_id": torch.tensor([b["seg_id"] for b in batch], dtype=torch.long),
    }


# -----------------------------------------------------------------------------
# DataLoader builders
# -----------------------------------------------------------------------------


def _make_loader(
    dataset: Dataset,
    shuffle: bool,
    collate_fn,
    cfg: ExperimentConfig,
) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and torch.cuda.is_available(),
    )


def build_cached_datasets(
    cfg: ExperimentConfig,
) -> tuple[CachedFeatureDataset, CachedFeatureDataset]:
    payload = load_feature_cache(features_file_for(cfg))
    _check_cache_matches_cfg(payload, cfg)

    features = payload["features"].float()
    group_ids = [int(g) for g in payload["group_id"]]
    roles = [str(r) for r in payload["role"]]
    track_ids = [str(t) for t in payload["track_id"]]
    seg_ids = [int(s) for s in payload["seg_id"]]

    train_groups, val_groups = split_group_ids(cfg, group_ids)
    train_idx = _indices_for_groups(group_ids, train_groups)
    val_idx = _indices_for_groups(group_ids, val_groups)

    if not train_idx or not val_idx:
        raise RuntimeError(
            f"Empty train or val split (train={len(train_idx)}, val={len(val_idx)}). "
            "Check val_fraction and cache contents."
        )

    train_ds = CachedFeatureDataset(features, group_ids, roles, track_ids, seg_ids, train_idx)
    val_ds = CachedFeatureDataset(features, group_ids, roles, track_ids, seg_ids, val_idx)
    return train_ds, val_ds


def build_online_datasets(
    cfg: ExperimentConfig,
) -> tuple[OnlineSegmentDataset, OnlineSegmentDataset]:
    segments = load_segments_table(cfg)
    group_ids = [int(g) for g in segments["group_id"]]

    train_groups, val_groups = split_group_ids(cfg, group_ids)
    train_idx = _indices_for_groups(group_ids, train_groups)
    val_idx = _indices_for_groups(group_ids, val_groups)

    if not train_idx or not val_idx:
        raise RuntimeError(
            f"Empty train or val split (train={len(train_idx)}, val={len(val_idx)})."
        )

    return (
        OnlineSegmentDataset(cfg, segments, train_idx),
        OnlineSegmentDataset(cfg, segments, val_idx),
    )


def build_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Return `(train_loader, val_loader)` for the configured augment mode."""
    if cfg.augment_mode == "online":
        train_ds, val_ds = build_online_datasets(cfg)
        collate_fn = collate_online
    else:
        train_ds, val_ds = build_cached_datasets(cfg)
        collate_fn = collate_cached

    return (
        _make_loader(train_ds, shuffle=True, collate_fn=collate_fn, cfg=cfg),
        _make_loader(val_ds, shuffle=False, collate_fn=collate_fn, cfg=cfg),
    )


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Propagate epoch to `OnlineSegmentDataset` (no-op for cached loaders)."""
    ds = loader.dataset
    if isinstance(ds, OnlineSegmentDataset):
        ds.set_epoch(epoch)
