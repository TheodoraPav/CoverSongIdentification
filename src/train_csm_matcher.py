"""Train the modular CSM + 2D CNN matcher (stage 2, after projection head).

Usage:
    python src/train_csm_matcher.py --config configs/<experiment>.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.csm_matcher import (  # noqa: E402
    csm_metrics_dict,
    evaluate_track_csm,
    evaluate_track_csm_diagonal,
    save_csm_matcher,
    save_csm_metrics,
    train_csm_matcher,
)
from src.dataset import build_dataloaders  # noqa: E402
from src.evaluate import collect_projected_embeddings, evaluate_loader, save_metrics  # noqa: E402
from src.model import build_projection_head  # noqa: E402
from src.utils import (  # noqa: E402
    checkpoint_path_for,
    csm_metrics_file_for,
    csm_matcher_checkpoint_path_for,
    get_logger,
    log_file_for,
    metrics_file_for,
    parse_config_arg,
    pick_device,
    set_global_seed,
)

LOGGER = get_logger("train_csm_matcher")


def _load_projection_head(cfg, device: torch.device) -> tuple[nn.Module, int | None]:
    ckpt_path = checkpoint_path_for(cfg)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Projection head checkpoint not found: {ckpt_path}. Run train.py first."
        )

    head = build_projection_head(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        head.load_state_dict(payload["state_dict"])
        best_epoch = payload.get("best_epoch")
    else:
        head.load_state_dict(payload)
        best_epoch = None

    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head, best_epoch


def run_csm_stage(cfg) -> dict:
    """Train CSM matcher and evaluate on val; returns combined metrics dict."""
    if not cfg.matcher.enabled:
        raise ValueError(
            "matcher.enabled is false. Set matcher.enabled: true in the YAML config."
        )

    device = pick_device()
    head, best_epoch = _load_projection_head(cfg, device)

    matcher = train_csm_matcher(cfg, head, device)
    ckpt_path = csm_matcher_checkpoint_path_for(cfg)
    save_csm_matcher(matcher, ckpt_path, cfg, extra={"projection_best_epoch": best_epoch})

    _, val_loader = build_dataloaders(cfg)
    data = collect_projected_embeddings(
        cfg, head, val_loader, device,
        epoch=int(best_epoch) if best_epoch is not None else 0,
    )

    diag_metrics = evaluate_track_csm_diagonal(data)
    csm_metrics = evaluate_track_csm(data, matcher, device)

    metrics = evaluate_loader(
        cfg, head, val_loader, device,
        None, None, None,
        epoch=int(best_epoch) if best_epoch is not None else 0,
    )
    metrics["experiment_name"] = cfg.experiment_name
    metrics.update(csm_metrics_dict(diag_metrics, prefix="track_csm_diag"))
    metrics.update(csm_metrics_dict(csm_metrics, prefix="track_csm"))
    metrics["csm_matcher_checkpoint"] = str(ckpt_path)

    save_metrics(metrics, metrics_file_for(cfg))
    save_csm_metrics(metrics, csm_metrics_file_for(cfg))

    LOGGER.info(
        "Val MRR | DTW=%.4f | CSM diagonal=%.4f | CSM+CNN=%.4f",
        metrics.get("track_dtw_mrr", float("nan")),
        diag_metrics["mrr"],
        csm_metrics["mrr"],
    )
    return metrics


def main() -> None:
    cfg = parse_config_arg("Train CSM + CNN matcher on frozen embeddings")
    get_logger("train_csm_matcher", log_file=log_file_for(cfg))
    set_global_seed(cfg.seed)
    run_csm_stage(cfg)


if __name__ == "__main__":
    main()
