"""PyTorch datasets and DataLoaders for training and evaluation.

Loads pooled vectors from `features.pt` produced by `extract_features.py`.

Public API:
    split_group_ids(cfg, all_group_ids) -> (train_groups, val_groups)
    CachedFeatureDataset
    build_dataloaders(cfg) -> (train_loader, val_loader)
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

from src.utils import (  # noqa: E402
    ExperimentConfig,
    features_file_for,
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
            "Run extract_features.py first."
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

    if cfg.augment != "none":
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


def build_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Return `(train_loader, val_loader)` for the offline cached features."""
    train_ds, val_ds = build_cached_datasets(cfg)
    return (
        _make_loader(train_ds, shuffle=True, collate_fn=collate_cached, cfg=cfg),
        _make_loader(val_ds, shuffle=False, collate_fn=collate_cached, cfg=cfg),
    )

