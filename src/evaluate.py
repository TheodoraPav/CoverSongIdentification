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
from src.model import build_projection_head, load_backbone  # noqa: E402
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
        backbone: nn.Module | None = None,
        processor: object | None = None,
        backbone_spec=None,
        epoch: int = 0,
) -> dict:
    head.eval()
    if backbone is not None:
        backbone.eval()

    z_list: list[torch.Tensor] = []
    group_ids: list[int] = []
    roles: list[str] = []

    for batch in loader:
        pooled = pooled_features_from_batch(
            batch, cfg, device, backbone, processor, backbone_spec, epoch,
        )
        z = head(pooled)
        z_list.append(z.cpu())
        group_ids.extend(int(g) for g in batch["group_id"].tolist())
        roles.extend(batch["role"])

    return {
        "z": torch.cat(z_list, dim=0),
        "group_id": group_ids,
        "role": roles,
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def compute_mrr_top5(
        z: torch.Tensor,
        group_ids: list[int],
        roles: list[str],
        top_k: int = 5,
) -> tuple[float, float]:
    z = F.normalize(z.float(), p=2, dim=1)
    gid = np.array(group_ids)
    role = np.array(roles)

    query_idx = np.where(role == "cover")[0]
    gallery_idx = np.where(role == "original")[0]

    if len(query_idx) == 0 or len(gallery_idx) == 0:
        return 0.0, 0.0

    sims = (z[query_idx] @ z[gallery_idx].T).cpu().numpy()
    gid_g = gid[gallery_idx]

    reciprocal_ranks: list[float] = []
    top_hits = 0

    for i, qi in enumerate(query_idx):
        order = np.argsort(-sims[i])
        ranked_gids = gid_g[order]
        hits = np.where(ranked_gids == gid[qi])[0]
        if hits.size == 0:
            reciprocal_ranks.append(0.0)
            continue
        rank = int(hits[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        if rank <= top_k:
            top_hits += 1

    mrr = float(np.mean(reciprocal_ranks))
    top5 = float(top_hits / len(query_idx))
    return mrr, top5


def compute_silhouette(z: torch.Tensor, group_ids: list[int]) -> float | None:
    labels = np.array(group_ids)
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) < len(unique) + 1:
        return None

    from sklearn.metrics import silhouette_score

    return float(silhouette_score(z.float().cpu().numpy(), labels, metric="cosine"))


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
        cfg, head, loader, device, backbone, processor, backbone_spec, epoch,
    )
    mrr, top5 = compute_mrr_top5(data["z"], data["group_id"], data["role"])
    sil = compute_silhouette(data["z"], data["group_id"])

    return {
        "experiment_name": cfg.experiment_name,
        "mrr": round(mrr, 6),
        "top5": round(top5, 6),
        "silhouette": round(sil, 6) if sil is not None else None,
        "n_segments": int(data["z"].shape[0]),
        "epoch": int(epoch),
    }


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

    head = build_projection_head(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        best_epoch = payload.get("best_epoch")
    else:
        # Backwards compatibility with older checkpoints (plain state_dict).
        state = payload
        best_epoch = None

    head.load_state_dict(state)
    head.eval()

    backbone = None
    processor = None
    backbone_spec = None
    if cfg.augment_mode == "online":
        backbone, processor, backbone_spec = load_backbone(
            cfg.backbone, cfg.backbone_checkpoint, device=device,
        )

    _, val_loader = build_dataloaders(cfg)
    metrics = evaluate_loader(
        cfg, head, val_loader, device,
        backbone, processor, backbone_spec, epoch=int(best_epoch) if best_epoch is not None else 0,
    )
    save_metrics(metrics, metrics_file_for(cfg))
    LOGGER.info(
        "MRR=%.4f Top-5=%.4f Silhouette=%s",
        metrics["mrr"], metrics["top5"], metrics["silhouette"],
    )


if __name__ == "__main__":
    main()