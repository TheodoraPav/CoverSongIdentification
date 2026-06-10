"""PyTorch datasets and DataLoaders for training and evaluation.

Loads pooled vectors from `features.pt` produced by `extract_features.py`.

`segment_pool_mode: fixed` (default)
    Every preprocessed segment is used every epoch.

`segment_pool_mode: dynamic`
    Preprocess/extract a capped duration pool per track. Each training epoch
    samples `segments_per_track` paired zone indices per group (original+cover).
    Validation uses a fixed, deterministic subset of the same size.

Public API:
    split_group_ids(cfg, all_group_ids) -> (train_groups, val_groups)
    CachedFeatureDataset
    GroupPairBatchSampler
    build_dataloaders(cfg) -> (train_loader, val_loader)
    set_epoch(loader, epoch)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    ExperimentConfig,
    features_file_for,
    get_logger,
    resolve_group_sampler_params,
)

LOGGER = get_logger("dataset")

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
    for key, expected in (
        ("backbone", cfg.backbone),
        ("sampling", cfg.sampling),
        ("segment_pool_mode", cfg.segment_pool_mode),
    ):
        found = payload.get(key)
        if found is not None and found != expected:
            raise ValueError(
                f"Cache {key}={found!r} does not match config {expected!r}. "
                "Re-run extract_features.py with this config."
            )

    if cfg.segment_pool_mode == "dynamic":
        found_max = payload.get("segment_pool_max")
        if found_max is not None and int(found_max) != cfg.segment_pool_max:
            raise ValueError(
                f"Cache segment_pool_max={found_max!r} does not match "
                f"config {cfg.segment_pool_max!r}. Re-run extract_features.py."
            )

    if cfg.augment != "none":
        cached_aug = payload.get("augment", "none")
        if cached_aug != cfg.augment:
            raise ValueError(
                f"Cache augment={cached_aug!r} does not match config "
                f"augment={cfg.augment!r}. Re-run extract_features.py."
            )


def _build_track_index(
    track_ids: list[str],
    seg_ids: list[int],
    indices: list[int],
) -> dict[str, list[int]]:
    """Map track_id -> cache row indices sorted by seg_id."""
    by_track: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_track[track_ids[idx]].append(idx)
    for tid in by_track:
        by_track[tid].sort(key=lambda i: seg_ids[i])
    return dict(by_track)


def _build_group_tracks(
    group_ids: list[int],
    track_ids: list[str],
    roles: list[str],
    indices: list[int],
) -> dict[int, dict[str, str]]:
    """Map group_id -> {role: track_id} within a split."""
    groups: dict[int, dict[str, str]] = defaultdict(dict)
    for idx in indices:
        gid = int(group_ids[idx])
        role = str(roles[idx]).lower()
        groups[gid][role] = track_ids[idx]
    return dict(groups)



def _select_fixed_eval_indices(
    cfg: ExperimentConfig,
    group_ids: list[int],
    roles: list[str],
    track_ids: list[str],
    indices: list[int],
    by_track: dict[str, list[int]],
) -> list[int]:
    """Deterministic evenly spaced zones for eval (same every epoch)."""
    group_tracks = _build_group_tracks(group_ids, track_ids, roles, indices)
    k = cfg.segments_per_track
    active: list[int] = []

    for gid in sorted(group_tracks):
        tracks = group_tracks[gid]
        if "original" not in tracks or "cover" not in tracks:
            continue
        orig_rows = by_track.get(tracks["original"], [])
        cover_rows = by_track.get(tracks["cover"], [])
        pool = min(len(orig_rows), len(cover_rows))
        if pool == 0:
            continue

        n_pick = min(k, pool)
        if pool <= n_pick:
            chosen = np.arange(pool, dtype=int)
        else:
            chosen = np.linspace(0, pool - 1, num=n_pick, dtype=int)
            chosen = np.unique(chosen)

        for pos in chosen.tolist():
            active.append(orig_rows[int(pos)])
            active.append(cover_rows[int(pos)])

    return active


def _select_dynamic_train_indices(
    cfg: ExperimentConfig,
    group_ids: list[int],
    roles: list[str],
    track_ids: list[str],
    indices: list[int],
    by_track: dict[str, list[int]],
    epoch: int,
) -> list[int]:
    """Sample paired zone indices per group for one training epoch."""
    group_tracks = _build_group_tracks(group_ids, track_ids, roles, indices)
    k = cfg.segments_per_track
    rng = np.random.default_rng(cfg.seed + int(epoch) * 1_000_003)
    active: list[int] = []

    for gid in sorted(group_tracks):
        tracks = group_tracks[gid]
        if "original" not in tracks or "cover" not in tracks:
            continue
        orig_rows = by_track.get(tracks["original"], [])
        cover_rows = by_track.get(tracks["cover"], [])
        pool = min(len(orig_rows), len(cover_rows))
        if pool == 0:
            continue

        n_pick = min(k, pool)
        chosen = rng.choice(pool, size=n_pick, replace=False)
        for pos in sorted(int(p) for p in chosen.tolist()):
            active.append(orig_rows[pos])
            active.append(cover_rows[pos])

    return active


class CachedFeatureDataset(Dataset):
    """One row per segment: pooled backbone vector + metadata."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        features: torch.Tensor,
        group_ids: list[int],
        roles: list[str],
        track_ids: list[str],
        seg_ids: list[int],
        indices: list[int],
        *,
        dynamic_train: bool = False,
        eval_fixed: bool = False,
    ) -> None:
        self.cfg = cfg
        self.features = features
        self.group_ids = group_ids
        self.roles = roles
        self.track_ids = track_ids
        self.seg_ids = seg_ids
        self.pool_indices = indices
        self.dynamic_train = dynamic_train
        self.eval_fixed = eval_fixed
        self.epoch = 0
        self.by_track = _build_track_index(track_ids, seg_ids, indices)
        self.active_indices = self._resolve_active_indices()

    def set_epoch(self, epoch: int) -> None:
        if not self.dynamic_train:
            return
        self.epoch = int(epoch)
        self.active_indices = self._resolve_active_indices()

    def _resolve_active_indices(self) -> list[int]:
        if self.eval_fixed:
            return _select_fixed_eval_indices(
                self.cfg,
                self.group_ids,
                self.roles,
                self.track_ids,
                self.pool_indices,
                self.by_track,
            )
        if self.dynamic_train:
            return _select_dynamic_train_indices(
                self.cfg,
                self.group_ids,
                self.roles,
                self.track_ids,
                self.pool_indices,
                self.by_track,
                self.epoch,
            )
        return list(self.pool_indices)

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.active_indices[i]
        return {
            "feature": self.features[idx],
            "group_id": int(self.group_ids[idx]),
            "role": str(self.roles[idx]),
            "track_id": str(self.track_ids[idx]),
            "seg_id": int(self.seg_ids[idx]),
        }


# -----------------------------------------------------------------------------
# Group-aware train batching
# -----------------------------------------------------------------------------


def _build_group_pair_index(
    dataset: CachedFeatureDataset,
) -> dict[int, dict[str, list[int]]]:
    """Map group_id -> role -> dataset positions for the active index set."""
    groups: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for pos, cache_idx in enumerate(dataset.active_indices):
        gid = int(dataset.group_ids[cache_idx])
        role = str(dataset.roles[cache_idx]).lower()
        if role in ("original", "cover"):
            groups[gid][role].append(pos)

    complete: dict[int, dict[str, list[int]]] = {}
    for gid, roles in groups.items():
        orig = roles.get("original", [])
        cover = roles.get("cover", [])
        if orig and cover:
            complete[gid] = {"original": orig, "cover": cover}
    return complete


def _build_group_slots(
    groups: dict[int, dict[str, list[int]]],
    segments_per_role: int,
    segments_per_track: int,
    rng: np.random.Generator,
) -> list[tuple[int, list[int], list[int]]]:
    """Chunk each group's active segments into paired original/cover slots."""
    slots: list[tuple[int, list[int], list[int]]] = []
    for gid in sorted(groups):
        orig = list(groups[gid]["original"])
        cover = list(groups[gid]["cover"])
        rng.shuffle(orig)
        rng.shuffle(cover)
        n = min(len(orig), len(cover), segments_per_track)
        for start in range(0, n, segments_per_role):
            o_chunk = orig[start : start + segments_per_role]
            c_chunk = cover[start : start + segments_per_role]
            if len(o_chunk) != len(c_chunk) or not o_chunk:
                continue
            slots.append((gid, o_chunk, c_chunk))
    rng.shuffle(slots)
    return slots


def _count_group_slots(
    groups: dict[int, dict[str, list[int]]],
    segments_per_role: int,
    segments_per_track: int,
) -> int:
    total = 0
    for roles in groups.values():
        n = min(len(roles["original"]), len(roles["cover"]), segments_per_track)
        if n > 0:
            total += (n + segments_per_role - 1) // segments_per_role
    return total


def iter_group_pair_batches(
    dataset: CachedFeatureDataset,
    *,
    groups_per_batch: int,
    segments_per_role: int,
    epoch: int,
    drop_last: bool = True,
) -> list[list[int]]:
    """Build one epoch of group-paired batches (for tests and the sampler).

    Every active segment (up to ``segments_per_track`` per role) appears once per
    epoch. Full batches contain ``groups_per_batch`` groups with
    ``segments_per_role`` segments per role; the final chunk for a group may be
    smaller when ``segments_per_track`` is not divisible by ``segments_per_role``.
    """
    groups = _build_group_pair_index(dataset)
    if not groups:
        return []

    rng = np.random.default_rng(dataset.cfg.seed + int(epoch) * 1_000_003)
    slots = _build_group_slots(
        groups,
        segments_per_role,
        dataset.cfg.segments_per_track,
        rng,
    )

    batches: list[list[int]] = []
    for start in range(0, len(slots), groups_per_batch):
        batch_slots = slots[start : start + groups_per_batch]
        if len(batch_slots) < groups_per_batch and drop_last:
            continue

        batch_indices: list[int] = []
        for _gid, o_chunk, c_chunk in batch_slots:
            batch_indices.extend(o_chunk)
            batch_indices.extend(c_chunk)

        if batch_indices:
            batches.append(batch_indices)

    return batches


class GroupPairBatchSampler(BatchSampler):
    """Batch sampler with paired original/cover segments per group_id.

    Each batch contains ``groups_per_batch`` song groups. For every group,
    ``segments_per_role`` segments are drawn from the original track and the
    same number from the cover track, so triplet / contrastive losses always
    see valid positives inside the batch.
    """

    def __init__(
        self,
        dataset: CachedFeatureDataset,
        *,
        groups_per_batch: int,
        segments_per_role: int,
        drop_last: bool = True,
    ) -> None:
        if groups_per_batch < 1 or segments_per_role < 1:
            raise ValueError(
                "groups_per_batch and segments_per_role must be >= 1, "
                f"got {groups_per_batch} and {segments_per_role}"
            )
        self.dataset = dataset
        self.groups_per_batch = int(groups_per_batch)
        self.segments_per_role = int(segments_per_role)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        for batch in iter_group_pair_batches(
            self.dataset,
            groups_per_batch=self.groups_per_batch,
            segments_per_role=self.segments_per_role,
            epoch=self.epoch,
            drop_last=self.drop_last,
        ):
            yield batch

    def __len__(self) -> int:
        groups = _build_group_pair_index(self.dataset)
        n_slots = _count_group_slots(
            groups,
            self.segments_per_role,
            self.dataset.cfg.segments_per_track,
        )
        if n_slots == 0:
            return 0
        if self.drop_last:
            return n_slots // self.groups_per_batch
        return (n_slots + self.groups_per_batch - 1) // self.groups_per_batch


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
    *,
    batch_sampler: BatchSampler | None = None,
) -> DataLoader:
    pin = cfg.training.pin_memory and torch.cuda.is_available()
    common = dict(
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=pin,
    )
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **common)
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=cfg.training.batch_size,
        **common,
    )


def _make_train_loader(
    train_ds: CachedFeatureDataset,
    cfg: ExperimentConfig,
) -> DataLoader:
    if not cfg.training.use_group_batch_sampler:
        return _make_loader(train_ds, shuffle=True, collate_fn=collate_cached, cfg=cfg)

    gpb, spr = resolve_group_sampler_params(cfg)
    sampler = GroupPairBatchSampler(
        train_ds,
        groups_per_batch=gpb,
        segments_per_role=spr,
        drop_last=cfg.training.group_sampler_drop_last,
    )
    LOGGER.info(
        "Group batch sampler enabled: %d groups/batch, %d segments/role/group "
        "(effective batch_size=%d)",
        gpb,
        spr,
        cfg.training.batch_size,
    )
    return _make_loader(
        train_ds,
        shuffle=False,
        collate_fn=collate_cached,
        cfg=cfg,
        batch_sampler=sampler,
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

    dynamic = cfg.segment_pool_mode == "dynamic"
    train_ds = CachedFeatureDataset(
        cfg, features, group_ids, roles, track_ids, seg_ids, train_idx,
        dynamic_train=dynamic,
    )
    val_ds = CachedFeatureDataset(
        cfg, features, group_ids, roles, track_ids, seg_ids, val_idx,
        eval_fixed=dynamic,
    )

    if dynamic:
        LOGGER.info(
            "Dynamic segment pool: train=%d pool rows, val=%d pool rows; "
            "using %d paired zones/train epoch, %d paired zones/eval.",
            len(train_idx),
            len(val_idx),
            len(train_ds),
            len(val_ds),
        )

    return train_ds, val_ds


def build_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Return `(train_loader, val_loader)` for the offline cached features."""
    train_ds, val_ds = build_cached_datasets(cfg)
    return (
        _make_train_loader(train_ds, cfg),
        _make_loader(val_ds, shuffle=False, collate_fn=collate_cached, cfg=cfg),
    )


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Propagate epoch to dynamic train datasets and group batch samplers."""
    ds = loader.dataset
    if isinstance(ds, CachedFeatureDataset):
        ds.set_epoch(epoch)

    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
