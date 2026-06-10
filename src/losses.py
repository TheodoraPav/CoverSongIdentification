"""Metric-learning losses for projection-head training."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def ntxent_loss(
        z: torch.Tensor,
        group_ids: torch.Tensor,
        temperature: float,
) -> torch.Tensor:
    """Multi-positive NT-Xent: positives are all same-group segments in the batch."""
    z = F.normalize(z, p=2, dim=1)
    b = z.size(0)
    if b < 2:
        return z.sum() * 0.0

    sim = torch.mm(z, z.t()) / temperature
    mask_self = torch.eye(b, device=z.device, dtype=torch.bool)
    gid = group_ids.view(-1, 1)
    same_group = (gid == gid.T) & ~mask_self

    losses = []
    for i in range(b):
        pos = same_group[i]
        if not pos.any():
            continue
        logits = sim[i].masked_fill(mask_self[i], float("-inf"))
        denom = torch.logsumexp(logits, dim=0)
        numer = torch.logsumexp(logits[pos], dim=0)
        losses.append(-(numer - denom))

    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def triplet_loss(
        z: torch.Tensor,
        group_ids: torch.Tensor,
        roles: list[str],
        margin: float,
        hard: bool = False,
) -> torch.Tensor:
    """Anchor = original, positive = same-group cover, negative = other group."""
    z = F.normalize(z, p=2, dim=1)
    dist = 1.0 - torch.mm(z, z.t())
    b = z.size(0)
    gid = group_ids.cpu().tolist()
    role = [str(r).lower() for r in roles]

    losses = []
    for i in range(b):
        if role[i] != "original":
            continue

        pos_idx = [j for j in range(b) if gid[j] == gid[i] and role[j] == "cover"]
        neg_idx = [j for j in range(b) if gid[j] != gid[i]]
        if not pos_idx or not neg_idx:
            continue

        j_pos = random.choice(pos_idx)
        if hard:
            j_neg = min(neg_idx, key=lambda j: dist[i, j].item())
        else:
            j_neg = random.choice(neg_idx)

        losses.append(F.relu(dist[i, j_pos] - dist[i, j_neg] + margin))

    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def proxy_anchor_loss(
        z: torch.Tensor,
        proxy_indices: torch.Tensor,
        proxies: torch.Tensor,
        *,
        alpha: float,
        delta: float,
) -> torch.Tensor:
    """Proxy-Anchor loss (Kim et al.) with one proxy per train group.

    Args:
        z: (B, D) segment embeddings (need not be pre-normalized).
        proxy_indices: (B,) row indices into ``proxies``.
        proxies: (C, D) L2-normalized proxy vectors.
        alpha: steepness / temperature-style scaling.
        delta: margin on cosine similarities.
    """
    z = F.normalize(z.float(), p=2, dim=1)
    proxies = F.normalize(proxies.float(), p=2, dim=1)
    sim = z @ proxies.T  # (B, C)

    batch_idx = torch.arange(sim.size(0), device=sim.device)
    pos_sim = sim[batch_idx, proxy_indices]

    pos_term = torch.log1p(torch.exp(-alpha * (pos_sim - delta)))

    neg_mask = torch.ones_like(sim, dtype=torch.bool)
    neg_mask[batch_idx, proxy_indices] = False
    neg_exp = torch.exp(alpha * (sim + delta)) * neg_mask
    neg_term = torch.log1p(neg_exp.sum(dim=1))

    return (pos_term + neg_term).mean()
