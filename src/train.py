"""Train the projection head on cached or live backbone features.

Usage:
    python src/train.py --config configs/<experiment>.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpointing import save_training_checkpoint  # noqa: E402
from src.dataset import build_dataloaders, get_train_group_ids, set_epoch  # noqa: E402
from src.evaluate import evaluate_loader, save_metrics  # noqa: E402
from src.extract_features import pooled_features_from_batch  # noqa: E402
from src.losses import ntxent_loss, proxy_anchor_loss, triplet_loss  # noqa: E402
from src.model import build_projection_head  # noqa: E402
from src.proxy_bank import ProxyBank  # noqa: E402
from src.utils import (  # noqa: E402
    ExperimentConfig,
    checkpoint_path_for,
    curves_plot_file_for,
    get_logger,
    history_file_for,
    log_file_for,
    metrics_file_for,
    parse_config_arg,
    pick_device,
    save_curves_plot,
    save_history,
    set_global_seed,
)

LOGGER = get_logger("train")


def compute_loss(
        z: torch.Tensor,
        batch: dict,
        cfg: ExperimentConfig,
        *,
        proxy_bank: ProxyBank | None = None,
) -> torch.Tensor:
    group_ids = batch["group_id"].to(z.device)
    roles = batch["role"]
    t = cfg.training

    if cfg.loss == "ntxent":
        return ntxent_loss(z, group_ids, t.ntxent_temperature)
    if cfg.loss == "triplet":
        return triplet_loss(z, group_ids, roles, t.triplet_margin, hard=False)
    if cfg.loss == "triplet_hard":
        return triplet_loss(z, group_ids, roles, t.triplet_margin, hard=True)
    if cfg.loss == "proxy_anchor":
        if proxy_bank is None:
            raise ValueError("proxy_bank is required when loss='proxy_anchor'")
        proxy_idx = proxy_bank.group_ids_to_indices(group_ids)
        return proxy_anchor_loss(
            z,
            proxy_idx,
            proxy_bank.all_proxies(),
            alpha=t.proxy_alpha,
            delta=t.proxy_delta,
        )
    raise ValueError(f"Unknown loss: {cfg.loss!r}")


def train_one_epoch(
        cfg: ExperimentConfig,
        head: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        *,
        proxy_bank: ProxyBank | None = None,
) -> float:
    head.train()
    if proxy_bank is not None:
        proxy_bank.train()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=f"train epoch {epoch}", leave=False):
        pooled = pooled_features_from_batch(batch, device)
        z = head(pooled)
        loss = compute_loss(z, batch, cfg, proxy_bank=proxy_bank)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / n_batches if n_batches else 0.0


def run_training(cfg: ExperimentConfig) -> dict:
    get_logger("train", log_file=log_file_for(cfg))

    set_global_seed(cfg.seed)
    device = pick_device()

    train_loader, val_loader = build_dataloaders(cfg)
    head = build_projection_head(cfg).to(device)

    proxy_bank: ProxyBank | None = None
    if cfg.loss == "proxy_anchor":
        train_group_ids = get_train_group_ids(cfg)
        proxy_bank = ProxyBank(train_group_ids, cfg.projection.output_dim).to(device)
        LOGGER.info(
            "Proxy-Anchor enabled with %d train proxies (alpha=%.2f, delta=%.2f)",
            len(train_group_ids),
            cfg.training.proxy_alpha,
            cfg.training.proxy_delta,
        )

    params = list(head.parameters())
    if proxy_bank is not None:
        params += list(proxy_bank.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    ckpt_path = checkpoint_path_for(cfg)
    best_mrr = -1.0
    best_metrics: dict | None = None
    best_epoch: int | None = None
    history: list[dict] = []
    patience = cfg.training.early_stopping_patience
    epochs_without_improvement = 0

    for epoch in range(1, cfg.training.epochs + 1):
        set_epoch(train_loader, epoch)

        train_loss = train_one_epoch(
            cfg, head, train_loader, optimizer, device, epoch,
            proxy_bank=proxy_bank,
        )
        LOGGER.info("Epoch %d | train_loss=%.4f", epoch, train_loss)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mrr": "",
            "val_top1": "",
            "val_top5": "",
            "val_silhouette": "",
        }

        if epoch % cfg.training.val_every == 0:
            metrics = evaluate_loader(
                cfg, head, val_loader, device,
                None, None, None, epoch=epoch,
            )
            LOGGER.info(
                "Epoch %d | MRR=%.4f Top-1=%.4f Top-5=%.4f Silhouette=%s",
                epoch, metrics["mrr"], metrics["top1"], metrics["top5"], metrics["silhouette"],
            )

            epoch_metrics["val_mrr"] = metrics["mrr"]
            epoch_metrics["val_top1"] = metrics["top1"]
            epoch_metrics["val_top5"] = metrics["top5"]
            epoch_metrics["val_silhouette"] = (
                metrics["silhouette"] if metrics["silhouette"] is not None else ""
            )

            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                best_metrics = metrics
                best_epoch = epoch
                save_training_checkpoint(
                    ckpt_path, head, epoch=epoch, proxy_bank=proxy_bank,
                )
                epochs_without_improvement = 0
            elif patience > 0:
                epochs_without_improvement += 1

        history.append(epoch_metrics)

        if patience > 0 and epochs_without_improvement >= patience:
            LOGGER.info(
                "Early stopping at epoch %d (no val MRR improvement for %d epochs; "
                "best epoch=%s, best MRR=%.4f)",
                epoch, patience, best_epoch, best_mrr,
            )
            break

    if best_metrics is None:
        LOGGER.warning("No validation ran; saving final weights.")
        save_training_checkpoint(
            ckpt_path, head, epoch=cfg.training.epochs, proxy_bank=proxy_bank,
        )
        best_metrics = evaluate_loader(
            cfg, head, val_loader, device,
            None, None, None, epoch=cfg.training.epochs,
        )
        best_epoch = cfg.training.epochs

    save_metrics(best_metrics, metrics_file_for(cfg))
    save_history(history, history_file_for(cfg))
    save_curves_plot(history, curves_plot_file_for(cfg), best_epoch=best_epoch)
    return best_metrics


def main() -> None:
    cfg = parse_config_arg("Train projection head for cover song identification")
    metrics = run_training(cfg)
    LOGGER.info("Done. Best MRR=%.4f Top-1=%.4f Top-5=%.4f", metrics["mrr"], metrics["top1"], metrics["top5"])


if __name__ == "__main__":
    main()
