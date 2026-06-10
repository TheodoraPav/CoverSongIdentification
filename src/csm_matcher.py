"""Cosine Similarity Matrix (CSM) + 2D CNN track matcher (modular stage 2).

Trained separately after the projection head is frozen. Replaces DTW scoring
at retrieval time with a learned binary classifier on CSM "images".

Public API:
    build_csm_matrix(seq_a, seq_b) -> np.ndarray
    CSMatchCNN
    build_csm_matcher(cfg) -> CSMatchCNN
    train_csm_matcher(cfg, head, device) -> CSMatchCNN
    evaluate_track_csm(data, matcher, device) -> dict
    load_csm_matcher(cfg, device) -> CSMatchCNN
    save_csm_matcher(matcher, path, cfg) -> None
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (  # noqa: E402
    CachedFeatureDataset,
    build_cached_datasets,
    collate_cached,
)
from src.track_sequences import build_track_sequences  # noqa: E402
from src.utils import (  # noqa: E402
    ExperimentConfig,
    csm_matcher_checkpoint_path_for,
    get_logger,
)

LOGGER = get_logger("csm_matcher")


# -----------------------------------------------------------------------------
# CSM construction
# -----------------------------------------------------------------------------


def build_csm_matrix(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two normalized segment sequences.

    Args:
        seq_a: (N, D) L2-normalized embeddings (query / cover).
        seq_b: (M, D) L2-normalized embeddings (gallery / original).

    Returns:
        (N, M) similarity matrix in [-1, 1].
    """
    if seq_a.ndim != 2 or seq_b.ndim != 2:
        raise ValueError(f"Expected 2D sequences, got {seq_a.shape} and {seq_b.shape}")
    return np.dot(seq_a, seq_b.T)


def csm_to_tensor(
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        size: int,
) -> torch.Tensor:
    """Build a single-channel CSM tensor resized to (1, size, size)."""
    csm = build_csm_matrix(seq_a, seq_b).astype(np.float32)
    t = torch.from_numpy(csm).unsqueeze(0).unsqueeze(0)  # (1, 1, N, M)
    if t.shape[-2:] != (size, size):
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0)  # (1, size, size)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class CSMatchCNN(nn.Module):
    """Lightweight 2D CNN binary matcher on CSM inputs."""

    def __init__(self, csm_size: int = 32) -> None:
        super().__init__()
        self.csm_size = int(csm_size)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B,)."""
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h).squeeze(-1)


def build_csm_matcher(cfg: ExperimentConfig) -> CSMatchCNN:
    return CSMatchCNN(csm_size=cfg.matcher.csm_size)


# -----------------------------------------------------------------------------
# Pair dataset for matcher training
# -----------------------------------------------------------------------------


class _CSMPairDataset(Dataset):
    def __init__(
            self,
            pairs: list[tuple[np.ndarray, np.ndarray]],
            labels: list[int],
            csm_size: int,
    ) -> None:
        self.pairs = pairs
        self.labels = labels
        self.csm_size = csm_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_q, seq_g = self.pairs[idx]
        x = csm_to_tensor(seq_q, seq_g, self.csm_size)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


def _make_csm_train_dataset(
        cfg: ExperimentConfig,
        track_sequences: dict[str, np.ndarray],
        track_info: dict[str, tuple[int, str]],
        *,
        rng: np.random.Generator,
) -> _CSMPairDataset:
    """Build positive and negative CSM pairs from train tracks only."""
    cover_tids = [tid for tid, (_, role) in track_info.items() if role == "cover"]
    orig_by_gid: dict[int, str] = {}
    for tid, (gid, role) in track_info.items():
        if role == "original":
            orig_by_gid[gid] = tid

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[int] = []

    all_orig_tids = [tid for tid, (_, role) in track_info.items() if role == "original"]
    if not cover_tids or not all_orig_tids:
        return _CSMPairDataset([], [], cfg.matcher.csm_size)

    negatives_per_positive = cfg.matcher.negatives_per_positive
    for cover_tid in cover_tids:
        gid, _ = track_info[cover_tid]
        pos_orig_tid = orig_by_gid.get(gid)
        if pos_orig_tid is None:
            continue
        seq_q = track_sequences[cover_tid]
        pairs.append((seq_q, track_sequences[pos_orig_tid]))
        labels.append(1)

        neg_candidates = [t for t in all_orig_tids if track_info[t][0] != gid]
        if not neg_candidates:
            continue
        n_neg = min(negatives_per_positive, len(neg_candidates))
        chosen = rng.choice(neg_candidates, size=n_neg, replace=False)
        for neg_tid in chosen:
            pairs.append((seq_q, track_sequences[str(neg_tid)]))
            labels.append(0)

    return _CSMPairDataset(pairs, labels, cfg.matcher.csm_size)


def _csm_train_loader_for_embeddings(
        cfg: ExperimentConfig,
        head: nn.Module,
        device: torch.device,
) -> DataLoader:
    """Collect projected embeddings on train split with fixed segment zones."""
    train_ds, _ = build_cached_datasets(cfg)
    if cfg.segment_pool_mode == "dynamic":
        train_ds = CachedFeatureDataset(
            train_ds.cfg,
            train_ds.features,
            train_ds.group_ids,
            train_ds.roles,
            train_ds.track_ids,
            train_ds.seg_ids,
            train_ds.pool_indices,
            eval_fixed=True,
        )
    loader = DataLoader(
        train_ds,
        shuffle=False,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_cached,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and torch.cuda.is_available(),
    )
    return loader


@torch.no_grad()
def _collect_train_track_sequences(
        cfg: ExperimentConfig,
        head: nn.Module,
        device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, str]]]:
    from src.evaluate import collect_projected_embeddings  # noqa: WPS433

    loader = _csm_train_loader_for_embeddings(cfg, head, device)
    data = collect_projected_embeddings(cfg, head, loader, device, epoch=0)
    track_sequences, track_info = build_track_sequences(data)
    return track_sequences, track_info


# -----------------------------------------------------------------------------
# Training / IO
# -----------------------------------------------------------------------------


def save_csm_matcher(
        matcher: CSMatchCNN,
        path: Path,
        cfg: ExperimentConfig,
        *,
        extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": matcher.state_dict(),
        "csm_size": cfg.matcher.csm_size,
        "matcher_config": {
            "epochs": cfg.matcher.epochs,
            "lr": cfg.matcher.lr,
            "batch_size": cfg.matcher.batch_size,
            "negatives_per_positive": cfg.matcher.negatives_per_positive,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    LOGGER.info("Saved CSM matcher to %s", path)


def load_csm_matcher(cfg: ExperimentConfig, device: torch.device) -> CSMatchCNN:
    path = csm_matcher_checkpoint_path_for(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSM matcher checkpoint not found: {path}. "
            "Run train_csm_matcher.py first."
        )
    matcher = build_csm_matcher(cfg)
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        matcher.load_state_dict(payload["state_dict"])
    else:
        matcher.load_state_dict(payload)
    matcher.to(device)
    matcher.eval()
    return matcher


def train_csm_matcher(
        cfg: ExperimentConfig,
        head: nn.Module,
        device: torch.device,
) -> CSMatchCNN:
    """Train the CSM CNN on train-split track pairs (frozen embeddings)."""
    head.eval()
    track_sequences, track_info = _collect_train_track_sequences(cfg, head, device)
    rng = np.random.default_rng(cfg.seed + 17_001)
    pair_ds = _make_csm_train_dataset(
        cfg, track_sequences, track_info, rng=rng,
    )
    if len(pair_ds) == 0:
        raise RuntimeError("No CSM training pairs could be built from the train split.")

    loader = DataLoader(
        pair_ds,
        batch_size=cfg.matcher.batch_size,
        shuffle=True,
        num_workers=0,
    )

    matcher = build_csm_matcher(cfg).to(device)
    optimizer = torch.optim.Adam(
        matcher.parameters(),
        lr=cfg.matcher.lr,
        weight_decay=cfg.matcher.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    LOGGER.info(
        "Training CSM matcher on %d pairs (%d positives approx.)",
        len(pair_ds),
        sum(1 for l in pair_ds.labels if l == 1),
    )

    for epoch in range(1, cfg.matcher.epochs + 1):
        matcher.train()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = matcher(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == y_batch).sum().item())
            total += int(y_batch.numel())

        acc = correct / max(total, 1)
        LOGGER.info(
            "CSM epoch %d/%d | loss=%.4f | acc=%.4f",
            epoch,
            cfg.matcher.epochs,
            total_loss / max(n_batches, 1),
            acc,
        )

    matcher.eval()
    return matcher


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def score_track_pair(
        matcher: CSMatchCNN,
        seq_query: np.ndarray,
        seq_gallery: np.ndarray,
        device: torch.device,
        csm_size: int,
) -> float:
    """Return P(match) for one (cover, original) track pair."""
    x = csm_to_tensor(seq_query, seq_gallery, csm_size).unsqueeze(0).to(device)
    logit = matcher(x)
    return float(torch.sigmoid(logit).item())


@torch.no_grad()
def evaluate_track_csm(
        data: dict,
        matcher: CSMatchCNN,
        device: torch.device,
        *,
        csm_size: int | None = None,
) -> dict:
    """Rank gallery originals per cover query using CSM+CNN match scores."""
    track_sequences, track_info = build_track_sequences(data)
    query_tids = [tid for tid, (_, role) in track_info.items() if role == "cover"]
    gallery_tids = [tid for tid, (_, role) in track_info.items() if role == "original"]

    if not query_tids or not gallery_tids:
        return {"mrr": 0.0, "top1": 0.0, "top5": 0.0}

    size = csm_size if csm_size is not None else matcher.csm_size
    query_gids = np.array([track_info[tid][0] for tid in query_tids])
    gallery_gids = np.array([track_info[tid][0] for tid in gallery_tids])

    scores = np.zeros((len(query_tids), len(gallery_tids)), dtype=np.float32)
    matcher.eval()
    for i, q_tid in enumerate(query_tids):
        seq_q = track_sequences[q_tid]
        for j, g_tid in enumerate(gallery_tids):
            scores[i, j] = score_track_pair(matcher, seq_q, track_sequences[g_tid], device, size)

    from src.evaluate import _rank_and_evaluate  # noqa: WPS433

    return _rank_and_evaluate(scores, query_gids, gallery_gids, descending=True)


def csm_metrics_dict(csm_metrics: dict) -> dict:
    return {
        "track_csm_mrr": round(csm_metrics["mrr"], 6),
        "track_csm_top1": round(csm_metrics["top1"], 6),
        "track_csm_top5": round(csm_metrics["top5"], 6),
    }


def save_csm_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Wrote CSM metrics to %s", path)
