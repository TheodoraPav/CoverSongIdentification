"""Re-evaluate all existing checkpoints to add track_voting metrics.

No re-training needed — this script loads each saved projection head
checkpoint, runs evaluation, and patches the metrics JSON files with
the new track_voting_mrr / track_voting_top1 / track_voting_top5 values.
"""

import os
import json
import torch
import glob
import yaml
from pathlib import Path

# Set paths
REPO = "/kaggle/working/CoverSongIdentification"
if not os.path.isdir(REPO):
    REPO = os.getcwd()  # local path fallback

import sys
sys.path.insert(0, REPO)

from src.utils import (
    load_config,
    checkpoint_path_for,
    metrics_file_for,
    pick_device,
    save_metrics,
)
from src.dataset import build_dataloaders
from src.model import build_projection_head
from src.evaluate import evaluate_loader
from src.checkpointing import load_head_from_checkpoint


# ---------------------------------------------------------------------------
# Heuristic: detect BN from experiment name
# ---------------------------------------------------------------------------
def _uses_batchnorm(name: str) -> bool:
    """Experiments with '_bn_' in their name were trained with BatchNorm."""
    return "_bn_" in name or name.endswith("_bn")



def main():
    results_dir = os.path.join(REPO, "results")
    if not os.path.isdir(results_dir):
        results_dir = "/kaggle/working/results"

    all_results_path = os.path.join(results_dir, "kaggle_all_results.json")
    if not os.path.isfile(all_results_path):
        print(f"Error: {all_results_path} not found.")
        return

    with open(all_results_path, encoding="utf-8") as f:
        all_results = json.load(f)

    device = pick_device()
    print(f"Device: {device}")

    base_config_path = os.path.join(REPO, "configs/baseline_mert_ntxent_kaggle.yaml")

    updated_count = 0

    for name, data in all_results.items():
        overrides = data.get("config_overrides", {})
        if not overrides:
            continue

        # Reconstruct paths
        manifest_hits = glob.glob(
            "/kaggle/input/**/audio_manifest.csv", recursive=True
        )
        if manifest_hits:
            manifest_path = manifest_hits[0]
            audio_root = os.path.dirname(os.path.dirname(manifest_path))
        else:
            manifest_path = "cover-dataset/data/audio_manifest.csv"
            audio_root = "cover-dataset"

        paths = {
            "manifest": manifest_path,
            "audio_root": audio_root,
            "segments_dir": "/kaggle/working/data_processed",
            "cache_dir": "/kaggle/working/cached_features",
            "checkpoints": "/kaggle/working/checkpoints",
            "results_dir": results_dir,
        }

        # Build config from base + overrides
        with open(base_config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Deep merge nested dicts so sub-keys aren't lost
        for section in ("training", "matcher", "projection"):
            if section in overrides and isinstance(overrides[section], dict):
                merged = dict(raw.get(section) or {})
                merged.update(overrides[section])
                overrides = {**overrides, section: merged}

        raw.update(overrides)
        raw["paths"] = paths
        raw["experiment_name"] = name

        # --- Auto-detect BN from experiment name ---
        bn = _uses_batchnorm(name)
        raw.setdefault("projection", {})
        raw["projection"]["batchnorm"] = bn


        # Write temp yaml so load_config can parse it
        tmp_yaml = os.path.join(results_dir, "_re_eval.yaml")
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(raw, f)

        cfg = load_config(tmp_yaml)

        ckpt_path = checkpoint_path_for(cfg)
        metrics_path = metrics_file_for(cfg)

        if ckpt_path.is_file():
            print(f"Checking {name}...")

            # Load current metrics – skip if track_voting_mrr already present
            has_voting = False
            current_metrics = {}
            if metrics_path.is_file():
                try:
                    with open(metrics_path, encoding="utf-8") as f:
                        current_metrics = json.load(f)
                    if "track_voting_mrr" in current_metrics:
                        has_voting = True
                except Exception as e:
                    print(f"  Error reading metrics for {name}: {e}")

            if has_voting:
                print(f" Already has voting metrics. Skipping.")
                continue

            print(f" Running evaluation for {name} (bn={bn})...")
            try:
                _, val_loader = build_dataloaders(cfg)
                head = build_projection_head(cfg).to(device)
                payload = torch.load(
                    ckpt_path, map_location=device, weights_only=False
                )
                best_epoch = load_head_from_checkpoint(head, payload, device)

                metrics = evaluate_loader(
                    cfg, head, val_loader, device, epoch=best_epoch
                )

                current_metrics.update(metrics)
                save_metrics(current_metrics, metrics_path)

                all_results[name].update(current_metrics)
                updated_count += 1
                print(f" Updated {name} successfully.")
            except Exception as e:
                print(f" Error evaluating {name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Checkpoint not found for {name} at {ckpt_path}")

    if updated_count > 0:
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(
            f"\n Done! Updated {updated_count} experiments in {all_results_path}."
        )
    else:
        print("\n No experiments needed re-evaluation.")


if __name__ == "__main__":
    main()
