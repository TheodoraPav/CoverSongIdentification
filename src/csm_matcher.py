"""Cosine Similarity Matrix (CSM) + 2D CNN track matcher (modular stage 2).

Trained separately after the projection head is frozen. Replaces DTW scoring
at retrieval time with a learned binary classifier on CSM "images".

Public API:
    build_csm_matrix(seq_a, seq_b) -> np.ndarray
    score_csm_diagonal(seq_q, seq_g) -> float
    CSMatchCNN
    train_csm_matcher(cfg, head, device) -> CSMatchCNN
    evaluate_track_csm(data, matcher, device) -> dict
    evaluate_track_csm_diagonal(data) -> dict
"""

from __future__ import annotations

import copy
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
    """Cosine similarity matrix between two normalized segment sequences."""
    if seq_a.ndim != 2 or seq_b.ndim != 2:
        raise ValueError(f"Expected 2D sequences, got {seq_a.shape} and {seq_b.shape}")
    return np.dot(seq_a, seq_b.T)


def csm_to_tensor(
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        size: int,
        *,
        resize: bool = False,
) -> torch.Tensor:
    """Build a single-channel CSM tensor of shape (1, size, size).

    ``resize=False`` (default): zero-pad to ``size`` — keeps sharp structure.
    ``resize=True``: bilinear upscale/downscale (legacy behaviour).
    """
    csm = build_csm_matrix(seq_a, seq_b).astype(np.float32)
    t = torch.from_numpy(csm).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    h, w = int(t.shape[-2]), int(t.shape[-1])

    if resize:
        if (h, w) != (size, size):
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    else:
        if h > size or w > size:
            t = t[:, :, :size, :size]
        pad_h = max(0, size - int(t.shape[-2]))
        pad_w = max(0, size - int(t.shape[-1]))
        if pad_h or pad_w:
            t = F.pad(t, (0, pad_w, 0, pad_h), value=0.0)

    return t.squeeze(0)  # (1, size, size)


def score_csm_diagonal(seq_query: np.ndarray, seq_gallery: np.ndarray) -> float:
    """Hand-crafted CSM score: mean similarity on the main diagonal.

    Segments are zone-aligned (same ``seg_id`` index), so a true cover–original
    pair often shows high values along the diagonal even with mild tempo drift.
    No trainable parameters — useful baseline before the CNN.
    """
    csm = build_csm_matrix(seq_query, seq_gallery)
    k = min(csm.shape)
    if k == 0:
        return 0.0
    return float(np.mean(np.diag(csm[:k, :k])))


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class CSMatchCNN(nn.Module):
    """Lightweight 2D CNN binary matcher on CSM inputs."""

    def __init__(self, csm_size: int = 10, *, csm_resize: bool = False) -> None:
        super().__init__()
        self.csm_size = int(csm_size)
        self.csm_resize = bool(csm_resize)
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
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h).squeeze(-1)


def build_csm_matcher(cfg: ExperimentConfig) -> CSMatchCNN:
    return CSMatchCNN(csm_size=cfg.matcher.csm_size, csm_resize=cfg.matcher.csm_resize)


# -----------------------------------------------------------------------------
# Pair dataset
# -----------------------------------------------------------------------------


class _CSMPairDataset(Dataset):
    def __init__(
            self,
            pairs: list[tuple[np.ndarray, np.ndarray]],
            labels: list[int],
            csm_size: int,
            *,
            resize: bool,
    ) -> None:
        self.pairs = pairs
        self.labels = labels
        self.csm_size = csm_size
        self.resize = resize

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_q, seq_g = self.pairs[idx]
        x = csm_to_tensor(seq_q, seq_g, self.csm_size, resize=self.resize)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


def _pick_negative_orig_tids(
        seq_q: np.ndarray,
        track_sequences: dict[str, np.ndarray],
        neg_candidates: list[str],
        n_neg: int,
        *,
        hard: bool,
        rng: np.random.Generator,
) -> list[str]:
    if n_neg <= 0 or not neg_candidates:
        return []
    if not hard:
        n_pick = min(n_neg, len(neg_candidates))
        return [str(t) for t in rng.choice(neg_candidates, size=n_pick, replace=False)]

    scored: list[tuple[float, str]] = []
    for tid in neg_candidates:
        mean_sim = float(np.mean(build_csm_matrix(seq_q, track_sequences[tid])))
        scored.append((mean_sim, str(tid)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in scored[: min(n_neg, len(scored))]]


def _make_csm_pair_dataset(
        cfg: ExperimentConfig,
        track_sequences: dict[str, np.ndarray],
        track_info: dict[str, tuple[int, str]],
        *,
        rng: np.random.Generator,
) -> _CSMPairDataset:
    """Build positive / negative CSM pairs from one track split."""
    cover_tids = [tid for tid, (_, role) in track_info.items() if role == "cover"]
    orig_by_gid: dict[int, str] = {}
    for tid, (gid, role) in track_info.items():
        if role == "original":
            orig_by_gid[int(gid)] = tid

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[int] = []

    all_orig_tids = [tid for tid, (_, role) in track_info.items() if role == "original"]
    if not cover_tids or not all_orig_tids:
        return _CSMPairDataset([], [], cfg.matcher.csm_size, resize=cfg.matcher.csm_resize)

    m = cfg.matcher
    for cover_tid in cover_tids:
        gid, _ = track_info[cover_tid]
        pos_orig_tid = orig_by_gid.get(int(gid))
        if pos_orig_tid is None:
            continue

        seq_q = track_sequences[cover_tid]
        seq_pos = track_sequences[pos_orig_tid]

        pairs.append((seq_q, seq_pos))
        labels.append(1)
        if m.symmetric_pairs:
            pairs.append((seq_pos, seq_q))
            labels.append(1)

        neg_candidates = [t for t in all_orig_tids if int(track_info[t][0]) != int(gid)]
        chosen = _pick_negative_orig_tids(
            seq_q,
            track_sequences,
            neg_candidates,
            m.negatives_per_positive,
            hard=m.hard_negatives,
            rng=rng,
        )
        for neg_tid in chosen:
            pairs.append((seq_q, track_sequences[neg_tid]))
            labels.append(0)

    return _CSMPairDataset(pairs, labels, m.csm_size, resize=m.csm_resize)


def _split_loader_for_embeddings(
        cfg: ExperimentConfig,
        *,
        train: bool,
) -> DataLoader:
    train_ds, val_ds = build_cached_datasets(cfg)
    dataset = train_ds if train else val_ds
    if cfg.segment_pool_mode == "dynamic":
        dataset = CachedFeatureDataset(
            dataset.cfg,
            dataset.features,
            dataset.group_ids,
            dataset.roles,
            dataset.track_ids,
            dataset.seg_ids,
            dataset.pool_indices,
            eval_fixed=True,
        )
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_cached,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and torch.cuda.is_available(),
    )


@torch.no_grad()
def _collect_track_sequences(
        cfg: ExperimentConfig,
        head: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, str]]]:
    from src.evaluate import collect_projected_embeddings  # noqa: WPS433

    data = collect_projected_embeddings(cfg, head, loader, device, epoch=0)
    return build_track_sequences(data)


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
        "csm_resize": cfg.matcher.csm_resize,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    LOGGER.info("Saved CSM matcher to %s", path)


def load_csm_matcher(cfg: ExperimentConfig, device: torch.device) -> CSMatchCNN:
    path = csm_matcher_checkpoint_path_for(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSM matcher checkpoint not found: {path}. Run train_csm_matcher.py first."
        )
    matcher = build_csm_matcher(cfg)
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        matcher.load_state_dict(payload["state_dict"])
        if "csm_resize" in payload:
            matcher.csm_resize = bool(payload["csm_resize"])
    else:
        matcher.load_state_dict(payload)
    matcher.to(device)
    matcher.eval()
    return matcher


def _positive_class_weight(labels: list[int], cfg: ExperimentConfig) -> float:
    if cfg.matcher.positive_class_weight is not None:
        return float(cfg.matcher.positive_class_weight)
    n_pos = max(1, sum(1 for y in labels if y == 1))
    n_neg = max(1, sum(1 for y in labels if y == 0))
    return float(n_neg) / float(n_pos)


def train_csm_matcher(
        cfg: ExperimentConfig,
        head: nn.Module,
        device: torch.device,
) -> CSMatchCNN:
    """Train CSM CNN on train pairs; early-stop on val MRR."""
    head.eval()
    m = cfg.matcher

    train_loader = _split_loader_for_embeddings(cfg, train=True)
    val_loader = _split_loader_for_embeddings(cfg, train=False)
    train_seq, train_info = _collect_track_sequences(cfg, head, train_loader, device)
    val_seq, val_info = _collect_track_sequences(cfg, head, val_loader, device)

    rng = np.random.default_rng(cfg.seed + 17_001)
    pair_ds = _make_csm_pair_dataset(cfg, train_seq, train_info, rng=rng)
    if len(pair_ds) == 0:
        raise RuntimeError("No CSM training pairs could be built from the train split.")

    val_data = {"track_sequences": val_seq, "track_info": val_info}

    loader = DataLoader(
        pair_ds,
        batch_size=m.batch_size,
        shuffle=True,
        num_workers=0,
    )

    matcher = build_csm_matcher(cfg).to(device)
    pos_weight = torch.tensor([_positive_class_weight(pair_ds.labels, cfg)], device=device)
    optimizer = torch.optim.Adam(
        matcher.parameters(),
        lr=m.lr,
        weight_decay=m.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_pos = sum(1 for y in pair_ds.labels if y == 1)
    n_neg = sum(1 for y in pair_ds.labels if y == 0)
    LOGGER.info(
        "CSM training pairs: %d (%d pos, %d neg) | pos_weight=%.2f | "
        "hard_neg=%s symmetric=%s pad=%s",
        len(pair_ds),
        n_pos,
        n_neg,
        float(pos_weight.item()),
        m.hard_negatives,
        m.symmetric_pairs,
        not m.csm_resize,
    )

    best_state = copy.deepcopy(matcher.state_dict())
    best_val_mrr = -1.0
    best_epoch = 0
    patience_left = m.early_stopping_patience

    for epoch in range(1, m.epochs + 1):
        matcher.train()
        total_loss = 0.0
        n_batches = 0

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

        val_metrics = _evaluate_track_csm_from_sequences(
            val_data["track_sequences"],
            val_data["track_info"],
            matcher,
            device,
            csm_size=m.csm_size,
            resize=m.csm_resize,
        )
        val_mrr = val_metrics["mrr"]
        improved = val_mrr > best_val_mrr
        if improved:
            best_val_mrr = val_mrr
            best_epoch = epoch
            best_state = copy.deepcopy(matcher.state_dict())
            patience_left = m.early_stopping_patience
        elif m.early_stopping_patience > 0:
            patience_left -= 1

        LOGGER.info(
            "CSM epoch %d/%d | loss=%.4f | val MRR=%.4f%s | patience=%d",
            epoch,
            m.epochs,
            total_loss / max(n_batches, 1),
            val_mrr,
            " *" if improved else "",
            patience_left,
        )

        if m.early_stopping_patience > 0 and patience_left <= 0:
            LOGGER.info("CSM early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    matcher.load_state_dict(best_state)
    matcher.eval()
    LOGGER.info("CSM best val MRR=%.4f at epoch %d", best_val_mrr, best_epoch)
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
        *,
        resize: bool = False,
) -> float:
    x = csm_to_tensor(seq_query, seq_gallery, csm_size, resize=resize).unsqueeze(0).to(device)
    return float(torch.sigmoid(matcher(x)).item())


def _evaluate_track_csm_from_sequences(
        track_sequences: dict[str, np.ndarray],
        track_info: dict[str, tuple[int, str]],
        matcher: CSMatchCNN | None,
        device: torch.device,
        *,
        csm_size: int,
        resize: bool,
        use_diagonal: bool = False,
) -> dict:
    from src.evaluate import _rank_and_evaluate  # noqa: WPS433

    query_tids = [tid for tid, (_, role) in track_info.items() if role == "cover"]
    gallery_tids = [tid for tid, (_, role) in track_info.items() if role == "original"]
    if not query_tids or not gallery_tids:
        return {"mrr": 0.0, "top1": 0.0, "top5": 0.0}

    query_gids = np.array([track_info[tid][0] for tid in query_tids])
    gallery_gids = np.array([track_info[tid][0] for tid in gallery_tids])
    scores = np.zeros((len(query_tids), len(gallery_tids)), dtype=np.float32)

    if matcher is not None:
        matcher.eval()

    for i, q_tid in enumerate(query_tids):
        seq_q = track_sequences[q_tid]
        for j, g_tid in enumerate(gallery_tids):
            seq_g = track_sequences[g_tid]
            if use_diagonal:
                scores[i, j] = score_csm_diagonal(seq_q, seq_g)
            else:
                scores[i, j] = score_track_pair(
                    matcher, seq_q, seq_g, device, csm_size, resize=resize,
                )

    return _rank_and_evaluate(scores, query_gids, gallery_gids, descending=True)


@torch.no_grad()
def evaluate_track_csm(
        data: dict,
        matcher: CSMatchCNN,
        device: torch.device,
        *,
        csm_size: int | None = None,
        resize: bool | None = None,
) -> dict:
    track_sequences, track_info = build_track_sequences(data)
    return _evaluate_track_csm_from_sequences(
        track_sequences,
        track_info,
        matcher,
        device,
        csm_size=csm_size if csm_size is not None else matcher.csm_size,
        resize=matcher.csm_resize if resize is None else resize,
    )


def evaluate_track_csm_diagonal(data: dict) -> dict:
    """Rank tracks using mean diagonal CSM similarity (no CNN)."""
    track_sequences, track_info = build_track_sequences(data)
    return _evaluate_track_csm_from_sequences(
        track_sequences,
        track_info,
        None,
        torch.device("cpu"),
        csm_size=0,
        resize=False,
        use_diagonal=True,
    )


def csm_metrics_dict(csm_metrics: dict, *, prefix: str = "track_csm") -> dict:
    return {
        f"{prefix}_mrr": round(csm_metrics["mrr"], 6),
        f"{prefix}_top1": round(csm_metrics["top1"], 6),
        f"{prefix}_top5": round(csm_metrics["top5"], 6),
    }


def save_csm_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Wrote CSM metrics to %s", path)
