"""Shared utilities: YAML config loader, seeding, logging, paths.

Public API (used by all other src/ modules):

    load_config(path)            -> ExperimentConfig
    parse_config_arg()           -> ExperimentConfig   (CLI helper)
    set_global_seed(seed)
    segment_seed(seed, gid, sid) -> int
    pick_device()                -> torch.device
    get_logger(name, log_file)
    segments_file_for(cfg)       -> Path
    cache_path_for(cfg)          -> Path
    features_file_for(cfg)       -> Path
    checkpoint_path_for(cfg)     -> Path
    metrics_file_for(cfg)        -> Path
    history_file_for(cfg)        -> Path
    curves_plot_file_for(cfg)    -> Path
    log_file_for(cfg)            -> Path
    umap_plot_file_for(cfg)      -> Path
    similarity_plot_file_for(cfg)-> Path
    silhouette_plot_file_for(cfg)-> Path
    resolve_audio_path(cfg, val) -> Path
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Allowed config enum values.
AUGMENTS = ("none", "time")
SAMPLINGS = ("random", "stratified", "beat", "mixed")
LOSSES = ("triplet", "triplet_hard", "ntxent", "proxy_anchor")
POOLS = ("mean", "max")
BACKBONES = ("mert", "mert_large")
EVAL_LEVELS = ("segment", "track_pool", "track_dtw")
SEGMENT_POOL_MODES = ("fixed", "dynamic")


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ProjectionConfig:
    input_dim: int | None = None  # auto-detected from backbone if None
    hidden_dim: int = 512
    output_dim: int = 128
    dropout: float = 0.3


@dataclass
class PathsConfig:
    manifest: str
    cache_dir: str
    checkpoints: str
    segments_dir: str = "data_processed"
    audio_root: str | None = None  # prepended to relative paths in the manifest
    results_dir: str = "results"


@dataclass
class MatcherConfig:
    """Stage-2 CSM + CNN matcher (trained after the projection head)."""

    enabled: bool = False
    csm_size: int = 10
    csm_resize: bool = False
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    negatives_per_positive: int = 4
    hard_negatives: bool = True
    symmetric_pairs: bool = True
    positive_class_weight: float | None = None
    early_stopping_patience: int = 5


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    val_every: int = 1
    val_fraction: float = 0.2
    triplet_margin: float = 0.3
    ntxent_temperature: float = 0.1
    proxy_alpha: float = 32.0
    proxy_delta: float = 0.1
    num_workers: int = 2
    pin_memory: bool = True
    early_stopping_patience: int = 12
    # Train batches with paired original/cover segments per group_id.
    # Default off so baseline YAMLs match earlier random-shuffle runs.
    use_group_batch_sampler: bool = False
    groups_per_batch: int | None = None
    segments_per_role_per_batch: int | None = None
    group_sampler_drop_last: bool = True


@dataclass
class ExperimentConfig:
    """Single source of truth for a run; maps 1:1 to a YAML in `configs/`."""

    experiment_name: str
    backbone: str
    backbone_checkpoint: str
    paths: PathsConfig
    sample_rate: int = 24000
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    augment: str = "none"
    sampling: str = "random"
    loss: str = "ntxent"
    pool: str = "mean"
    segments_per_track: int = 5
    segment_seconds: float = 5.0
    segment_pool_mode: str = "fixed"
    segment_pool_max: int = 20
    seed: int = 42
    eval_level: str = "segment"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)

    def validate(self) -> None:
        enum_fields = (
            ("backbone", self.backbone, BACKBONES),
            ("augment", self.augment, AUGMENTS),
            ("sampling", self.sampling, SAMPLINGS),
            ("loss", self.loss, LOSSES),
            ("pool", self.pool, POOLS),
            ("eval_level", self.eval_level, EVAL_LEVELS),
            ("segment_pool_mode", self.segment_pool_mode, SEGMENT_POOL_MODES),
        )
        for name, value, allowed in enum_fields:
            _ensure_in(name, value, allowed)

        if self.segments_per_track < 1:
            raise ValueError("segments_per_track must be >= 1")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be > 0")
        if self.segment_pool_mode == "dynamic" and self.segment_pool_max < 1:
            raise ValueError("segment_pool_max must be >= 1 when segment_pool_mode='dynamic'")
        if not (0.0 < self.training.val_fraction < 1.0):
            raise ValueError("training.val_fraction must be in (0, 1)")
        if self.training.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be >= 0")
        if self.training.use_group_batch_sampler:
            resolve_group_sampler_params(self)
        if self.matcher.csm_size < 4:
            raise ValueError("matcher.csm_size must be >= 4")
        if self.matcher.epochs < 1:
            raise ValueError("matcher.epochs must be >= 1")
        if self.matcher.negatives_per_positive < 1:
            raise ValueError("matcher.negatives_per_positive must be >= 1")
        if self.matcher.early_stopping_patience < 0:
            raise ValueError("matcher.early_stopping_patience must be >= 0")


def _ensure_in(name: str, value: object, allowed: Iterable[str]) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {name}={value!r}. Allowed: {sorted(allowed)}")


def resolve_group_sampler_params(cfg: ExperimentConfig) -> tuple[int, int]:
    """Resolve (groups_per_batch, segments_per_role_per_batch) for group batching.

    Requires ``batch_size == 2 * groups_per_batch * segments_per_role_per_batch``.
    """
    t = cfg.training
    bs = t.batch_size
    gpb = t.groups_per_batch
    spr = t.segments_per_role_per_batch

    if gpb is None and spr is None:
        # Largest segments/role (<= segments_per_track) that fills batch_size exactly.
        spr_cap = min(cfg.segments_per_track, bs // 2)
        gpb, spr = 1, 1
        for spr_candidate in range(spr_cap, 0, -1):
            if bs % (2 * spr_candidate) == 0:
                spr = spr_candidate
                gpb = bs // (2 * spr_candidate)
                break
    elif gpb is None:
        if spr < 1:
            raise ValueError("training.segments_per_role_per_batch must be >= 1")
        gpb = bs // (2 * spr)
    elif spr is None:
        if gpb < 1:
            raise ValueError("training.groups_per_batch must be >= 1")
        spr = bs // (2 * gpb)
    else:
        if gpb < 1 or spr < 1:
            raise ValueError(
                "training.groups_per_batch and segments_per_role_per_batch must be >= 1"
            )

    if gpb < 1 or spr < 1:
        raise ValueError(
            f"Could not derive valid group sampler params from batch_size={bs}. "
            "Set training.groups_per_batch and/or training.segments_per_role_per_batch."
        )

    expected = 2 * gpb * spr
    if expected != bs:
        raise ValueError(
            f"training.batch_size={bs} must equal "
            f"2 * groups_per_batch * segments_per_role_per_batch "
            f"(2 * {gpb} * {spr} = {expected}) when use_group_batch_sampler=True"
        )
    return gpb, spr


# -----------------------------------------------------------------------------
# YAML loading
# -----------------------------------------------------------------------------


def load_config(path: str | os.PathLike) -> ExperimentConfig:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a YAML mapping, got {type(raw)}")

    paths_raw = raw.get("paths") or {}
    for required in ("manifest", "cache_dir", "checkpoints"):
        if required not in paths_raw:
            raise ValueError(f"`paths.{required}` is required in {p.name}")
    paths = PathsConfig(**paths_raw)
    projection = ProjectionConfig(**(raw.get("projection") or {}))
    training = TrainingConfig(**(raw.get("training") or {}))
    matcher = MatcherConfig(**(raw.get("matcher") or {}))

    cfg = ExperimentConfig(
        experiment_name=raw["experiment_name"],
        backbone=raw["backbone"],
        backbone_checkpoint=raw["backbone_checkpoint"],
        paths=paths,
        sample_rate=raw.get("sample_rate", 24000),
        projection=projection,
        augment=raw.get("augment", "none"),
        sampling=raw.get("sampling", "random"),
        loss=raw.get("loss", "ntxent"),
        pool=raw.get("pool", "mean"),
        segments_per_track=raw.get("segments_per_track", 5),
        segment_seconds=raw.get("segment_seconds", 5.0),
        segment_pool_mode=raw.get("segment_pool_mode", "fixed"),
        segment_pool_max=raw.get("segment_pool_max", 20),
        seed=raw.get("seed", 42),
        eval_level=raw.get("eval_level", "segment"),
        training=training,
        matcher=matcher,
    )
    cfg.validate()
    return cfg


def parse_config_arg(description: str = "Cover Song Identification") -> ExperimentConfig:
    """CLI helper: `python src/<script>.py --config configs/foo.yaml`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    args, _ = parser.parse_known_args()
    return load_config(args.config)


# -----------------------------------------------------------------------------
# Seeding
# -----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def segment_seed(global_seed: int, group_id: int | str, seg_id: int) -> int:
    """Deterministic per-segment seed.

    Same `(group_id, seg_id)` always yields the same int; used to apply
    identical augmentation to original + cover (coupled positives).
    """
    digest = hashlib.blake2s(
        f"{group_id}|{seg_id}".encode("utf-8"), digest_size=8
    ).hexdigest()
    return (int(global_seed) + int(digest, 16)) & 0x7FFFFFFF


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def get_logger(
    name: str,
    log_file: str | os.PathLike | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """stdout handler + optional file handler. Re-callable without duplicates."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_stdout = any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        for h in logger.handlers
    )
    if not has_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        has_file = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_path
            for h in logger.handlers
        )
        if not has_file:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------


def pool_size_for_duration(duration: float, seg_len: float, max_pool: int) -> int:
    """Number of non-overlapping fixed-length segments a track can fit (capped)."""
    if seg_len <= 0:
        raise ValueError("seg_len must be > 0")
    n = max(1, int(duration // seg_len))
    return min(n, max_pool)


def segments_file_for(cfg: ExperimentConfig) -> Path:
    """Segments CSV path; dynamic mode stores the full capped pool separately."""
    base = Path(cfg.paths.segments_dir) / cfg.sampling
    if cfg.segment_pool_mode == "dynamic":
        return base / f"pool_{cfg.segment_pool_max}" / "segments.csv"
    return base / "segments.csv"


def cache_path_for(cfg: ExperimentConfig) -> Path:
    """Feature cache directory; dynamic pools use a dedicated subdirectory."""
    base = Path(cfg.paths.cache_dir) / cfg.backbone / cfg.pool / cfg.augment / cfg.sampling
    if cfg.segment_pool_mode == "dynamic":
        return base / f"pool_{cfg.segment_pool_max}"
    return base


def features_file_for(cfg: ExperimentConfig) -> Path:
    return cache_path_for(cfg) / "features.pt"


def experiment_id_for(cfg: ExperimentConfig) -> str:
    """Generate a unique experiment ID combining all configuration parameters."""
    base = (
        f"{cfg.backbone}_{cfg.loss}_{cfg.augment}_{cfg.sampling}_seed{cfg.seed}_"
        f"{cfg.pool}_{cfg.eval_level}"
    )
    if cfg.segment_pool_mode == "dynamic":
        return f"{base}_dyn{cfg.segments_per_track}from{cfg.segment_pool_max}"
    return base


def checkpoint_path_for(cfg: ExperimentConfig) -> Path:
    """`checkpoints/{backbone}/{experiment_id}_best_head.pt`."""
    return Path(cfg.paths.checkpoints) / cfg.backbone / f"{experiment_id_for(cfg)}_best_head.pt"


def csm_matcher_checkpoint_path_for(cfg: ExperimentConfig) -> Path:
    """`checkpoints/{backbone}/{experiment_id}_csm_matcher.pt`."""
    return (
        Path(cfg.paths.checkpoints) / cfg.backbone
        / f"{experiment_id_for(cfg)}_csm_matcher.pt"
    )


def csm_metrics_file_for(cfg: ExperimentConfig) -> Path:
    """`results/metrics/{experiment_id}_csm.json`."""
    return Path(cfg.paths.results_dir) / "metrics" / f"{experiment_id_for(cfg)}_csm.json"


def metrics_file_for(cfg: ExperimentConfig) -> Path:
    """`results/metrics/{experiment_id}.json`."""
    return Path(cfg.paths.results_dir) / "metrics" / f"{experiment_id_for(cfg)}.json"


def history_file_for(cfg: ExperimentConfig) -> Path:
    """`results/history/{experiment_id}_history.csv`."""
    return Path(cfg.paths.results_dir) / "history" / f"{experiment_id_for(cfg)}_history.csv"


def curves_plot_file_for(cfg: ExperimentConfig) -> Path:
    """`results/figures/{experiment_id}_curves.png`."""
    return Path(cfg.paths.results_dir) / "figures" / f"{experiment_id_for(cfg)}_curves.png"


def log_file_for(cfg: ExperimentConfig) -> Path:
    """`results/logs/{experiment_id}.log`."""
    return Path(cfg.paths.results_dir) / "logs" / f"{experiment_id_for(cfg)}.log"


def umap_plot_file_for(cfg: ExperimentConfig) -> Path:
    """`results/figures/{experiment_id}_umap.png`."""
    return Path(cfg.paths.results_dir) / "figures" / f"{experiment_id_for(cfg)}_umap.png"


def similarity_plot_file_for(cfg: ExperimentConfig) -> Path:
    """`results/figures/{experiment_id}_similarity.png`."""
    return Path(cfg.paths.results_dir) / "figures" / f"{experiment_id_for(cfg)}_similarity.png"


def silhouette_plot_file_for(cfg: ExperimentConfig) -> Path:
    """`results/figures/{experiment_id}_silhouette.png`."""
    return Path(cfg.paths.results_dir) / "figures" / f"{experiment_id_for(cfg)}_silhouette.png"


def _collapse_duplicate_path_parts(path: Path) -> Path:
    """Collapse consecutive duplicate directory names (e.g. ``data/data/audio``)."""
    parts = list(path.parts)
    if not parts:
        return path
    out: list[str] = []
    for part in parts:
        if out and out[-1] == part:
            continue
        out.append(part)
    return Path(*out)


def _strip_redundant_path_prefix(rel: Path, base: Path) -> Path:
    """Drop a leading segment when it repeats ``base.name`` (manifest/base mismatch)."""
    if rel.parts and base.name and rel.parts[0] == base.name:
        return Path(*rel.parts[1:])
    return rel


def resolve_audio_path(cfg: ExperimentConfig, value: str) -> Path:
    """Resolve a manifest `audio_path` (Windows-style relative) to a full Path.

    Manifest rows often store ``data/audio/<id>.wav`` while ``audio_root`` or the
    manifest directory is already ``.../data``. Without normalization that yields
    ``.../data/data/audio/...``. Absolute paths stored in ``segments.csv`` are
    normalized the same way.
    """
    p = Path(str(value).replace("\\", "/"))
    if p.is_absolute():
        return _collapse_duplicate_path_parts(p)

    if cfg.paths.audio_root:
        base = Path(cfg.paths.audio_root)
    else:
        base = Path(cfg.paths.manifest).resolve().parent

    p = _strip_redundant_path_prefix(p, base)
    return _collapse_duplicate_path_parts(base / p)


def save_history(history: list[dict], path: Path) -> None:
    logger = logging.getLogger("utils")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys()) if history else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    logger.info("Wrote training history to %s", path)


def save_curves_plot(history: list[dict], path: Path, best_epoch: int | None = None) -> None:
    logger = logging.getLogger("utils")
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]

    val_epochs = [h["epoch"] for h in history if h["val_mrr"] != ""]
    val_mrr = [h["val_mrr"] for h in history if h["val_mrr"] != ""]
    val_top1 = [h["val_top1"] for h in history if h.get("val_top1", "") != ""]
    val_top5 = [h["val_top5"] for h in history if h["val_top5"] != ""]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_loss = "tab:red"
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Train Loss", color=color_loss, fontsize=11, fontweight="bold")
    line1 = ax1.plot(epochs, losses, color=color_loss, label="Train Loss", linewidth=2.0)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color_mrr = "tab:blue"
    color_top1 = "tab:orange"
    color_top5 = "tab:green"
    ax2.set_ylabel("Validation Metrics", color="black", fontsize=11, fontweight="bold")
    line2 = ax2.plot(val_epochs, val_mrr, color=color_mrr, marker="o", markersize=5, label="Val MRR", linewidth=1.5)
    line_top1 = ax2.plot(val_epochs, val_top1, color=color_top1, marker="^", markersize=4, label="Val Top-1", linewidth=1.5)
    line3 = ax2.plot(val_epochs, val_top5, color=color_top5, marker="s", markersize=4, label="Val Top-5", linewidth=1.5)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(0.0, 1.05)

    lines = line1 + line2 + line_top1 + line3
    
    if best_epoch is not None:
        best_line = ax1.axvline(x=best_epoch, color="tab:purple", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Best Epoch ({best_epoch})")
        lines.append(best_line)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, shadow=False, facecolor="white", edgecolor="lightgray")

    plt.title("Training Loss and Validation Retrieval Performance", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved curves plot to %s", path)
