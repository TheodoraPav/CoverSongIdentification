"""Learnable class proxies for Proxy-Anchor metric learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyBank(nn.Module):
    """One L2-normalized proxy vector per train ``group_id``."""

    def __init__(self, train_group_ids: list[int] | set[int], output_dim: int) -> None:
        super().__init__()
        unique = sorted({int(g) for g in train_group_ids})
        if not unique:
            raise ValueError("train_group_ids must not be empty")

        self.output_dim = int(output_dim)
        self.group_ids: list[int] = unique
        self.group_id_to_index = {gid: idx for idx, gid in enumerate(unique)}
        self.proxies = nn.Embedding(len(unique), self.output_dim)
        nn.init.xavier_uniform_(self.proxies.weight)

    def all_proxies(self) -> torch.Tensor:
        """Return all proxies as (C, D) L2-normalized."""
        return F.normalize(self.proxies.weight, p=2, dim=1)

    def group_ids_to_indices(self, group_ids: torch.Tensor) -> torch.Tensor:
        """Map batch group_ids to proxy row indices."""
        ids = group_ids.detach().cpu().tolist()
        try:
            indices = [self.group_id_to_index[int(g)] for g in ids]
        except KeyError as exc:
            raise KeyError(
                "Batch contains group_id outside train proxy bank. "
                "Proxy-Anchor training must use train-split groups only."
            ) from exc
        return torch.tensor(indices, device=group_ids.device, dtype=torch.long)
