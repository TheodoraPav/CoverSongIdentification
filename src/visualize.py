"""Generate visualizations for metric-learning embedding space.

Provides UMAP cluster plots with linked positive pairs, Before vs. After (Projection Head)
pairwise cosine similarity matrix comparisons, and Silhouette score progression.

Usage:
    python src/visualize.py --config configs/<experiment>.yaml
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # doesn't open a pop-up window, saves the plot directly to a file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataloaders  # noqa: E402
from src.extract_features import pooled_features_from_batch  # noqa: E402
from src.model import build_projection_head  # noqa: E402
from src.utils import (  # noqa: E402
    ExperimentConfig,
    checkpoint_path_for,
    get_logger,
    history_file_for,
    parse_config_arg,
    pick_device,
    set_global_seed,
    umap_plot_file_for,
    similarity_plot_file_for,
    silhouette_plot_file_for,
)

LOGGER = get_logger("visualize")


# -----------------------------------------------------------------------------
# Embedding Collection
# -----------------------------------------------------------------------------

@torch.no_grad()
def collect_embeddings_before_after(
        cfg: ExperimentConfig,
        head: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> dict:
    head.eval()
    raw_list: list[torch.Tensor] = []
    z_list: list[torch.Tensor] = []
    group_ids: list[int] = []
    roles: list[str] = []
    track_ids: list[str] = []

    for batch in tqdm(loader, desc="Extracting visualizer features"):
        # extract pooled backbone features (raw)
        pooled = pooled_features_from_batch(
            batch, device,
        )
        # extract projected embeddings (after projection head)
        z = head(pooled)
        raw_list.append(pooled.cpu())
        z_list.append(z.cpu())
        group_ids.extend(int(g) for g in batch["group_id"].tolist())
        roles.extend(batch["role"]) # original or cover
        if "track_id" in batch:
            track_ids.extend(batch["track_id"])

    return {
        "raw": torch.cat(raw_list, dim=0),
        "z": torch.cat(z_list, dim=0),
        "group_id": group_ids,
        "role": roles,
        "track_id": track_ids,
    }


def average_by_track(
        embeddings: torch.Tensor,
        track_ids: list[str],
        group_ids: list[int],
        roles: list[str],
) -> tuple[torch.Tensor, list[str], list[int], list[str]]:
    # average segment embeddings to form a single track-level embedding vector for each song (micro level -> macro level)
    unique_tracks = sorted(list(set(track_ids)))
    
    avg_embeddings = []
    out_track_ids = []
    out_group_ids = []
    out_roles = []
    
    track_to_idx = {}
    for i, tid in enumerate(track_ids):
        if tid not in track_to_idx:
            track_to_idx[tid] = []
        track_to_idx[tid].append(i)
        
    for tid in unique_tracks:
        idxs = track_to_idx[tid]
        avg_emb = embeddings[idxs].mean(dim=0)
        avg_embeddings.append(avg_emb)
        
        # metadata values are identical across segments of the same track
        out_track_ids.append(tid)
        out_group_ids.append(group_ids[idxs[0]])
        out_roles.append(roles[idxs[0]])
        
    return torch.stack(avg_embeddings, dim=0), out_track_ids, out_group_ids, out_roles


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_umap_space(
        embeddings: torch.Tensor,
        group_ids: list[int],
        roles: list[str],
        seed: int,
        path: Path,
        title: str,
) -> None:
    # reduce embeddings to 2D UMAP space and highlight selected song groups with dashed links
    LOGGER.info("Reducing embedding space with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
    reduced = reducer.fit_transform(embeddings.numpy())

    gids = np.array(group_ids)
    uniq_gids = np.unique(gids)

    # count segments (or tracks) in each group
    group_counts = {gid: np.sum(gids == gid) for gid in uniq_gids}
    
    # filter groups that have at least 2 tracks (safety check to ensure both original + cover exist)
    valid_groups = [g for g in uniq_gids if group_counts[g] >= 2]
    
    # sort groups by count in descending order and select top 6
    sorted_groups = sorted(valid_groups, key=group_counts.get, reverse=True)
    highlight_groups = sorted_groups[:6]

    fig, ax = plt.subplots(figsize=(10, 8))

    # plot background (other song groups) in light, transparent gray
    bg_mask = ~np.isin(gids, highlight_groups)
    ax.scatter(
        reduced[bg_mask, 0], reduced[bg_mask, 1],
        c="lightgray", alpha=0.4, s=25, label="Other Songs", edgecolors="none"
    )

    # use distinct color cycle and shapes for highlighted cover groups
    colors = plt.cm.tab10(np.linspace(0, 1, len(highlight_groups)))
    markers = ["o", "s", "^", "D", "v", "p"]

    for idx, gid in enumerate(highlight_groups):
        mask = (gids == gid)
        group_reduced = reduced[mask]
        group_roles = np.array(roles)[mask]

        c = colors[idx]
        m = markers[idx % len(markers)]

        # plot validation covers
        cov_mask = (group_roles == "cover")
        if np.any(cov_mask):
            ax.scatter(
                group_reduced[cov_mask, 0], group_reduced[cov_mask, 1],
                color=c, marker=m, s=60, alpha=0.85, edgecolors="black", linewidths=0.6,
                label=f"Group {gid} (Covers)"
            )

        # plot validation original query
        orig_mask = (group_roles == "original")
        if np.any(orig_mask):
            ax.scatter(
                group_reduced[orig_mask, 0], group_reduced[orig_mask, 1],
                color=c, marker=m, s=160, alpha=1.0, edgecolors="black", linewidths=2.0,
                label=f"Group {gid} (Original Query)"
            )

            # draw lines from the original to its cover songs
            orig_coord = group_reduced[orig_mask][0]
            for cov_coord in group_reduced[cov_mask]:
                ax.plot(
                    [orig_coord[0], cov_coord[0]], [orig_coord[1], cov_coord[1]],
                    color=c, linestyle="--", linewidth=1.2, alpha=0.7
                )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # clean up plot axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved UMAP scatter plot to %s", path)


def plot_similarity_comparison(
        raw_embeddings: torch.Tensor,
        z_embeddings: torch.Tensor,
        group_ids: list[int],
        path: Path,
) -> None:
    # plot side-by-side pairwise similarity matrix heatmaps showing before vs. after learning
    LOGGER.info("Computing cosine similarity matrices for comparative heatmap...")
    
    # normalize features -> dot product equal to cosine similarity
    raw_norm = F.normalize(raw_embeddings.float(), p=2, dim=1)
    z_norm = F.normalize(z_embeddings.float(), p=2, dim=1)

    gids = np.array(group_ids)
    sorted_indices = np.argsort(gids)

    # filter to a subset of the first 8 song groups to keep the heatmap detailed and legible
    uniq_gids = np.unique(gids[sorted_indices])
    selected_groups = uniq_gids[:8]
    subset_indices = [idx for idx in sorted_indices if gids[idx] in selected_groups]

    sub_raw = raw_norm[subset_indices]
    sub_z = z_norm[subset_indices]
    sub_gids = gids[subset_indices]

    # compute pairwise similarity matrices
    sim_raw = (sub_raw @ sub_raw.T).numpy()
    sim_z = (sub_z @ sub_z.T).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # heatmap before projection
    im1 = ax1.imshow(sim_raw, cmap="viridis", vmin=0.0, vmax=1.0)
    ax1.set_title("Before Projection (Raw Backbone Features)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Song Segments (Grouped)")
    ax1.set_ylabel("Song Segments (Grouped)")

    # heatmap after projection
    im2 = ax2.imshow(sim_z, cmap="viridis", vmin=0.0, vmax=1.0)
    ax2.set_title("After Projection (Trainable Metric Space)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Song Segments (Grouped)")

    # draw white dotted lines to mark distinct song group boundaries
    boundaries = np.where(sub_gids[:-1] != sub_gids[1:])[0]
    for b in boundaries:
        for ax in (ax1, ax2):
            ax.axhline(b + 0.5, color="white", linewidth=0.8, linestyle=":")
            ax.axvline(b + 0.5, color="white", linewidth=0.8, linestyle=":")

    # add shared colorbar
    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    fig.colorbar(im2, cax=cbar_ax, label="Cosine Similarity Value")

    plt.suptitle("Embedding Cosine Similarity Matrix Comparison (Before vs. After)", fontsize=14, fontweight="bold", y=0.96)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved side-by-side similarity heatmaps to %s", path)


def plot_silhouette_progression(history_path: Path, path: Path) -> None:
    # plot validation Silhouette score progression over epochs from the training CSV history
    if not history_path.is_file():
        LOGGER.warning("CSV History not found at %s; skipping silhouette plot.", history_path)
        return

    LOGGER.info("Reading history CSV to plot Silhouette progression...")
    df = pd.read_csv(history_path)
    df = df.dropna(subset=["val_silhouette"])
    
    if df.empty:
        LOGGER.warning("No validation Silhouette values found in history CSV; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["epoch"], df["val_silhouette"],
        color="tab:purple", marker="d", markersize=6, linewidth=2.0, label="Val Silhouette"
    )
    
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Silhouette Score (Cosine Distance)", fontsize=11, fontweight="bold")
    ax.set_title("Validation Embedding Cluster Cohesion Trend (Silhouette Score)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, edgecolor="lightgray")

    # Clean up plot axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved Silhouette progression chart to %s", path)


# -----------------------------------------------------------------------------
# Main Visualizer Logic
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = parse_config_arg("Generate UMAP, Cosine Similarity Matrices, and Silhouette charts")
    set_global_seed(cfg.seed)
    device = pick_device()

    ckpt_path = checkpoint_path_for(cfg)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {ckpt_path}. "
            "Please run train.py first before visualizing results."
        )

    LOGGER.info("Building data loaders & loading model weights...")
    _, val_loader = build_dataloaders(cfg)
    
    head = build_projection_head(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        # Backwards compatibility with older checkpoints (plain state_dict).
        state = payload

    head.load_state_dict(state)
    head.eval()

    # collect embeddings before vs after
    embeddings = collect_embeddings_before_after(
        cfg, head, val_loader, device
    )

    # average embeddings to track level for a clean UMAP visualization
    track_z, _, track_gids, track_roles = average_by_track(
        embeddings=embeddings["z"],
        track_ids=embeddings["track_id"],
        group_ids=embeddings["group_id"],
        roles=embeddings["role"],
    )

    # render UMAP scatter plot at the Track Level
    plot_umap_space(
        embeddings=track_z,
        group_ids=track_gids,
        roles=track_roles,
        seed=cfg.seed,
        path=umap_plot_file_for(cfg),
        title=f"2D UMAP Track-Level Embedding Manifold ({cfg.experiment_name})"
    )

    # render pairwise similarity matrices (Before vs. After)
    plot_similarity_comparison(
        raw_embeddings=embeddings["raw"],
        z_embeddings=embeddings["z"],
        group_ids=embeddings["group_id"],
        path=similarity_plot_file_for(cfg)
    )

    # render Silhouette score progression
    plot_silhouette_progression(
        history_path=history_file_for(cfg),
        path=silhouette_plot_file_for(cfg)
    )

    LOGGER.info("All figures generated and saved successfully!")


if __name__ == "__main__":
    main()
