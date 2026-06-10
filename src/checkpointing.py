"""Save/load projection-head checkpoints (with optional ProxyBank metadata)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.proxy_bank import ProxyBank


def save_training_checkpoint(
        path: Path,
        head: nn.Module,
        *,
        epoch: int | None = None,
        proxy_bank: ProxyBank | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "head_state_dict": head.state_dict(),
        "best_epoch": int(epoch) if epoch is not None else None,
    }
    if proxy_bank is not None:
        payload["proxy_state_dict"] = proxy_bank.state_dict()
        payload["train_group_ids"] = list(proxy_bank.group_ids)
    torch.save(payload, path)


def load_head_from_checkpoint(
        head: nn.Module,
        payload: object,
        device: torch.device | str,
) -> int | None:
    """Load projection-head weights; supports legacy and proxy-anchor formats."""
    if isinstance(payload, dict):
        if "head_state_dict" in payload:
            head.load_state_dict(payload["head_state_dict"])
            best_epoch = payload.get("best_epoch")
        elif "state_dict" in payload:
            head.load_state_dict(payload["state_dict"])
            best_epoch = payload.get("best_epoch")
        else:
            head.load_state_dict(payload)
            best_epoch = None
    else:
        head.load_state_dict(payload)
        best_epoch = None

    head.to(device)
    head.eval()
    return int(best_epoch) if best_epoch is not None else None
