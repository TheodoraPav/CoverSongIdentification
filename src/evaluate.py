"""Validation metrics: MRR, Top-5, Silhouette.

Retrieval protocol:
    Query   = validation segments with role ``cover``.
    Gallery = validation segments with role ``original``.
    Rank gallery by cosine similarity to each query embedding.
    A hit is the gallery original whose ``group_id`` matches the cover query.

Usage:
    python src/evaluate.py --config configs/<experiment>.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataloaders  # noqa: E402
from src.extract_features import pooled_features_from_batch  # noqa: E402
from src.checkpointing import load_head_from_checkpoint  # noqa: E402
from src.model import build_projection_head  # noqa: E402
from src.track_sequences import build_track_sequences  # noqa: E402
from src.utils import (  # noqa: E402
    ExperimentConfig,
    checkpoint_path_for,
    get_logger,
    log_file_for,
    metrics_file_for,
    parse_config_arg,
    pick_device,
    set_global_seed,
)

LOGGER = get_logger("evaluate")


# -----------------------------------------------------------------------------
# Embedding collection
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_projected_embeddings(
        cfg: ExperimentConfig,
        head: nn.Module,
        loader: DataLoader,
        device: torch.device,
        epoch: int = 0,
) -> dict:
    head.eval()
    z_list: list[torch.Tensor] = []
    group_ids: list[int] = []
    roles: list[str] = []
    track_ids: list[str] = []
    seg_ids: list[int] = []

    for batch in loader:
        pooled = pooled_features_from_batch(
            batch, device)
        z = head(pooled)
        z_list.append(z.cpu())
        group_ids.extend(int(g) for g in batch["group_id"].tolist())
        roles.extend(batch["role"])
        track_ids.extend(batch["track_id"])
        seg_ids.extend(int(s) for s in batch["seg_id"].tolist())

    return {
        "z": torch.cat(z_list, dim=0),
        "group_id": group_ids,
        "role": roles,
        "track_id": track_ids,
        "seg_id": seg_ids,
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def compute_mrr_top1_top5(
        z: torch.Tensor,
        group_ids: list[int],
        roles: list[str],
) -> tuple[float, float, float]:
    z = F.normalize(z.float(), p=2, dim=1)
    gid = np.array(group_ids)
    role = np.array(roles)

    query_idx = np.where(role == "cover")[0]
    gallery_idx = np.where(role == "original")[0]

    if len(query_idx) == 0 or len(gallery_idx) == 0:
        return 0.0, 0.0, 0.0

    sims = (z[query_idx] @ z[gallery_idx].T).cpu().numpy()
    gid_g = gid[gallery_idx]

    reciprocal_ranks: list[float] = []
    top1_hits = 0
    top5_hits = 0

    for i, qi in enumerate(query_idx):
        order = np.argsort(-sims[i])
        ranked_gids = gid_g[order]
        hits = np.where(ranked_gids == gid[qi])[0]
        if hits.size == 0:
            reciprocal_ranks.append(0.0)
            continue
        rank = int(hits[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        if rank <= 1:
            top1_hits += 1
        if rank <= 5:
            top5_hits += 1

    mrr = float(np.mean(reciprocal_ranks))
    top1 = float(top1_hits / len(query_idx))
    top5 = float(top5_hits / len(query_idx))
    return mrr, top1, top5


def compute_silhouette(z: torch.Tensor, group_ids: list[int]) -> float | None:
    labels = np.array(group_ids)
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) < len(unique) + 1:
        return None

    from sklearn.metrics import silhouette_score

    return float(silhouette_score(z.float().cpu().numpy(), labels, metric="cosine"))


def compute_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Compute the Dynamic Time Warping (DTW) distance using Cosine Distance.
    
    Args:
        s1: Normalized sequence array of shape (N, D)
        s2: Normalized sequence array of shape (M, D)
        
    Returns:
        The DTW distance normalized by path length.
    """
    n, d = s1.shape
    m, _ = s2.shape
    
    # Cosine distance grid (already normalized)
    dist_matrix = 1.0 - np.dot(s1, s2.T)
    
    # DP table
    dp = np.zeros((n, m))
    dp[0, 0] = dist_matrix[0, 0]
    
    for i in range(1, n):
        dp[i, 0] = dp[i-1, 0] + dist_matrix[i, 0]
        
    for j in range(1, m):
        dp[0, j] = dp[0, j-1] + dist_matrix[0, j]
        
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = dist_matrix[i, j] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
            
    return float(dp[-1, -1] / (n + m))


def _rank_and_evaluate(
        scores: np.ndarray,
        query_gids: np.ndarray,
        gallery_gids: np.ndarray,
        descending: bool = True,
) -> dict:
    """Rank gallery elements for each query and calculate MRR, Top-1, and Top-5 scores."""
    reciprocal_ranks = []
    top1_hits = 0
    top5_hits = 0
    
    for i in range(len(query_gids)):
        order = np.argsort(-scores[i]) if descending else np.argsort(scores[i])
        ranked_gids = gallery_gids[order]
        hits = np.where(ranked_gids == query_gids[i])[0]
        if hits.size == 0:
            reciprocal_ranks.append(0.0)
            continue
        rank = int(hits[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        if rank <= 1:
            top1_hits += 1
        if rank <= 5:
            top5_hits += 1
            
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "top1": float(top1_hits / len(query_gids)),
        "top5": float(top5_hits / len(query_gids)),
    }


def evaluate_track_level(data: dict) -> tuple[dict, dict]:
    """Compute track-level evaluation metrics: Pooling (Early Fusion) and Sequence DTW.
    
    Args:
        data: A dictionary containing z, group_id, role, track_id, and seg_id.
            
    Returns:
        A tuple of dictionaries: (pool_metrics, dtw_metrics)
    """
    track_sequences, track_info = build_track_sequences(data)

    query_tids = [tid for tid, info in track_info.items() if info[1] == "cover"]
    gallery_tids = [tid for tid, info in track_info.items() if info[1] == "original"]
    
    if len(query_tids) == 0 or len(gallery_tids) == 0:
        empty = {"mrr": 0.0, "top1": 0.0, "top5": 0.0}
        return empty, empty
        
    query_gids = np.array([track_info[tid][0] for tid in query_tids])
    gallery_gids = np.array([track_info[tid][0] for tid in gallery_tids])
    
    # --- Track Pooling evaluation (Early Fusion) ---
    query_pooled = np.stack([track_sequences[tid].mean(axis=0) for tid in query_tids], axis=0)
    gallery_pooled = np.stack([track_sequences[tid].mean(axis=0) for tid in gallery_tids], axis=0)
    
    qp_norm = query_pooled / (np.linalg.norm(query_pooled, axis=1, keepdims=True) + 1e-9)
    gp_norm = gallery_pooled / (np.linalg.norm(gallery_pooled, axis=1, keepdims=True) + 1e-9)
    
    sims = np.dot(qp_norm, gp_norm.T)
    pool_metrics = _rank_and_evaluate(sims, query_gids, gallery_gids, descending=True)
    
    # --- Track DTW evaluation (Late Fusion sequence alignment) ---
    dtw_distances = np.zeros((len(query_tids), len(gallery_tids)))
    for i, q_tid in enumerate(query_tids):
        s_q = track_sequences[q_tid]
        for j, g_tid in enumerate(gallery_tids):
            s_g = track_sequences[g_tid]
            dtw_distances[i, j] = compute_dtw_distance(s_q, s_g)
            
    dtw_metrics = _rank_and_evaluate(dtw_distances, query_gids, gallery_gids, descending=False)
    
    return pool_metrics, dtw_metrics


def evaluate_loader(
        cfg: ExperimentConfig,
        head: nn.Module,
        loader: DataLoader,
        device: torch.device,
        backbone: nn.Module | None = None,
        processor: object | None = None,
        backbone_spec=None,
        epoch: int = 0,
) -> dict:
    data = collect_projected_embeddings(
        cfg, head, loader, device, epoch,
    )
    
    # Segment-level (baseline) metrics
    mrr, top1, top5 = compute_mrr_top1_top5(data["z"], data["group_id"], data["role"])
    sil = compute_silhouette(data["z"], data["group_id"])

    # Track-level metrics
    pool_metrics, dtw_metrics = evaluate_track_level(data)
    
    track_pool_mrr = pool_metrics["mrr"]
    track_pool_top1 = pool_metrics["top1"]
    track_pool_top5 = pool_metrics["top5"]
    
    track_dtw_mrr = dtw_metrics["mrr"]
    track_dtw_top1 = dtw_metrics["top1"]
    track_dtw_top5 = dtw_metrics["top5"]

    # Show formatted comparison table
    seg_sel = " [Selected]" if cfg.eval_level == "segment" else ""
    pool_sel = " [Selected]" if cfg.eval_level == "track_pool" else ""
    dtw_sel = " [Selected]" if cfg.eval_level == "track_dtw" else ""
    
    LOGGER.info("=" * 75)
    LOGGER.info("EVALUATION LEVEL COMPARISON (mrr / top1 / top5)")
    LOGGER.info("=" * 75)
    LOGGER.info("Segment-Level (Baseline) : %.4f / %.4f / %.4f%s", mrr, top1, top5, seg_sel)
    LOGGER.info("Track-Level Mean Pooling : %.4f / %.4f / %.4f%s", track_pool_mrr, track_pool_top1, track_pool_top5, pool_sel)
    LOGGER.info("Track-Level Sequence DTW : %.4f / %.4f / %.4f%s", track_dtw_mrr, track_dtw_top1, track_dtw_top5, dtw_sel)

    track_csm_mrr = None
    track_csm_top1 = None
    track_csm_top5 = None
    if cfg.matcher.enabled:
        from src.csm_matcher import (  # noqa: WPS433 - optional stage-2 matcher
            csm_metrics_dict,
            evaluate_track_csm,
            load_csm_matcher,
        )
        from src.utils import csm_matcher_checkpoint_path_for  # noqa: WPS433

        csm_ckpt = csm_matcher_checkpoint_path_for(cfg)
        if csm_ckpt.is_file():
            matcher = load_csm_matcher(cfg, device)
            csm_metrics = evaluate_track_csm(data, matcher, device)
            track_csm_mrr = csm_metrics["mrr"]
            track_csm_top1 = csm_metrics["top1"]
            track_csm_top5 = csm_metrics["top5"]
            LOGGER.info(
                "Track-Level CSM + CNN      : %.4f / %.4f / %.4f",
                track_csm_mrr,
                track_csm_top1,
                track_csm_top5,
            )
        else:
            LOGGER.info(
                "Track-Level CSM + CNN      : skipped (no checkpoint at %s)",
                csm_ckpt,
            )

    LOGGER.info("=" * 75)

    # Map primary metrics based on configured eval_level
    if cfg.eval_level == "segment":
        primary_mrr = mrr
        primary_top1 = top1
        primary_top5 = top5
    elif cfg.eval_level == "track_pool":
        primary_mrr = track_pool_mrr
        primary_top1 = track_pool_top1
        primary_top5 = track_pool_top5
    elif cfg.eval_level == "track_dtw":
        primary_mrr = track_dtw_mrr
        primary_top1 = track_dtw_top1
        primary_top5 = track_dtw_top5
    else:
        primary_mrr = mrr
        primary_top1 = top1
        primary_top5 = top5

    metrics = {
        "experiment_name": cfg.experiment_name,
        "mrr": round(primary_mrr, 6),
        "top1": round(primary_top1, 6),
        "top5": round(primary_top5, 6),
        "silhouette": round(sil, 6) if sil is not None else None,
        "n_segments": int(data["z"].shape[0]),
        "epoch": int(epoch),
        "eval_level_config": cfg.eval_level,
        
        # Detailed segment metrics
        "segment_mrr": round(mrr, 6),
        "segment_top1": round(top1, 6),
        "segment_top5": round(top5, 6),
        
        # Detailed track pool metrics
        "track_pool_mrr": round(track_pool_mrr, 6),
        "track_pool_top1": round(track_pool_top1, 6),
        "track_pool_top5": round(track_pool_top5, 6),
        
        # Detailed track dtw metrics
        "track_dtw_mrr": round(track_dtw_mrr, 6),
        "track_dtw_top1": round(track_dtw_top1, 6),
        "track_dtw_top5": round(track_dtw_top5, 6),
    }

    if track_csm_mrr is not None:
        metrics.update(csm_metrics_dict({
            "mrr": track_csm_mrr,
            "top1": track_csm_top1,
            "top5": track_csm_top5,
        }))

    return metrics


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Wrote metrics to %s", path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = parse_config_arg("Evaluate a trained projection head on the val split")
    get_logger("evaluate", log_file=log_file_for(cfg))
    
    set_global_seed(cfg.seed)
    device = pick_device()

    ckpt_path = checkpoint_path_for(cfg)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    head = build_projection_head(cfg)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_epoch = load_head_from_checkpoint(head, payload, device)

    _, val_loader = build_dataloaders(cfg)
    metrics = evaluate_loader(
        cfg, head, val_loader, device,
        None, None, None, epoch=int(best_epoch) if best_epoch is not None else 0,
    )
    save_metrics(metrics, metrics_file_for(cfg))
    LOGGER.info(
        "MRR=%.4f Top-1=%.4f Top-5=%.4f Silhouette=%s",
        metrics["mrr"], metrics["top1"], metrics["top5"], metrics["silhouette"],
    )


if __name__ == "__main__":
    main()