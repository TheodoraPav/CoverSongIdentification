"""Re-evaluate all existing checkpoints to add track_voting metrics.

No re-training needed — this script loads each saved projection head
checkpoint, runs evaluation, and patches the metrics JSON files with
the new track_voting_mrr / track_voting_top1 / track_voting_top5 values.
"""

import os
import json
import shutil
import torch
import glob
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-discover repo & Kaggle paths
# ---------------------------------------------------------------------------
REPO = "/kaggle/working/CoverSongIdentification"
if not os.path.isdir(REPO):
    REPO = os.getcwd()

import sys
sys.path.insert(0, REPO)

from src.utils import load_config, pick_device
from src.dataset import build_dataloaders
from src.model import build_projection_head
from src.evaluate import evaluate_loader, save_metrics
from src.checkpointing import load_head_from_checkpoint


def _uses_batchnorm(name: str) -> bool:
    """Experiments with '_bn_' in their name were trained with BatchNorm."""
    return "_bn_" in name or name.endswith("_bn")


def _find_input_results_dir() -> str | None:
    """Find the results/ dir inside the Kaggle input dataset."""
    hits = glob.glob("/kaggle/input/**/results/kaggle_all_results.json", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    return None


def main():
    # --- Locate kaggle_all_results.json ---
    # On Kaggle the results live in the read-only input dataset.
    # We copy them to /kaggle/working/results so we can update them.
    input_results_dir = _find_input_results_dir()

    output_dir = "/kaggle/working/results"
    if not os.path.isdir("/kaggle/working"):
        # Local fallback
        output_dir = os.path.join(REPO, "results")

    os.makedirs(output_dir, exist_ok=True)

    if input_results_dir:
        src_json = os.path.join(input_results_dir, "kaggle_all_results.json")
        dst_json = os.path.join(output_dir, "kaggle_all_results.json")
        if not os.path.isfile(dst_json):
            shutil.copy2(src_json, dst_json)
            print(f"Copied kaggle_all_results.json to {output_dir}")
        all_results_path = dst_json
    else:
        # Fallback: check REPO/results or output_dir
        for candidate in [
            os.path.join(output_dir, "kaggle_all_results.json"),
            os.path.join(REPO, "results", "kaggle_all_results.json"),
        ]:
            if os.path.isfile(candidate):
                all_results_path = candidate
                break
        else:
            print("Error: kaggle_all_results.json not found anywhere.")
            return

    with open(all_results_path, encoding="utf-8") as f:
        all_results = json.load(f)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Results JSON: {all_results_path}")
    print(f"Input results dir: {input_results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Total experiments: {len(all_results)}")
    print()

    base_config_path = os.path.join(REPO, "configs/baseline_mert_ntxent_kaggle.yaml")

    # --- Discover paths ---
    manifest_hits = glob.glob("/kaggle/input/**/audio_manifest.csv", recursive=True)
    if manifest_hits:
        manifest_path = manifest_hits[0]
        audio_root = os.path.dirname(os.path.dirname(manifest_path))
    else:
        manifest_path = "cover-dataset/data/audio_manifest.csv"
        audio_root = "cover-dataset"

    updated_count = 0

    for name, data in all_results.items():
        overrides = data.get("config_overrides", {})
        if not overrides:
            continue

        # ---- Build the config ----
        with open(base_config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Deep merge nested dicts
        for section in ("training", "matcher", "projection"):
            if section in overrides and isinstance(overrides[section], dict):
                merged = dict(raw.get(section) or {})
                merged.update(overrides[section])
                overrides = {**overrides, section: merged}

        raw.update(overrides)
        raw["experiment_name"] = name

        # Auto-detect BN from experiment name
        bn = _uses_batchnorm(name)
        raw.setdefault("projection", {})
        raw["projection"]["batchnorm"] = bn

        # Set paths — results_dir points to the INPUT dataset (read-only)
        # so checkpoint_path_for and metrics_file_for resolve correctly
        raw["paths"] = {
            "manifest": manifest_path,
            "audio_root": audio_root,
            "segments_dir": "/kaggle/working/data_processed",
            "cache_dir": "/kaggle/working/cached_features",
            "checkpoints": "/kaggle/working/checkpoints",
            "results_dir": input_results_dir or output_dir,
        }

        # Write temp yaml for load_config
        tmp_yaml = os.path.join(output_dir, "_re_eval.yaml")
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(raw, f)

        cfg = load_config(tmp_yaml)

        # Checkpoint lives in the input dataset: results/{name}/best_head.pt
        if input_results_dir:
            ckpt_path = Path(input_results_dir) / name / "best_head.pt"
        else:
            ckpt_path = Path(cfg.paths.results_dir) / name / "best_head.pt"

        if not ckpt_path.is_file():
            print(f"  SKIP {name}: no checkpoint at {ckpt_path}")
            continue

        print(f"Checking {name}...")

        # Check if track_voting_mrr already exists
        if "track_voting_mrr" in data:
            print(f"  Already has voting metrics. Skipping.")
            continue

        print(f"  Running evaluation (bn={bn})...")
        try:
            _, val_loader = build_dataloaders(cfg)
            head = build_projection_head(cfg).to(device)
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            best_epoch = load_head_from_checkpoint(head, payload, device)

            metrics = evaluate_loader(cfg, head, val_loader, device, epoch=best_epoch)

            # Save updated metrics to output dir
            out_exp_dir = Path(output_dir) / name
            out_exp_dir.mkdir(parents=True, exist_ok=True)
            out_metrics = out_exp_dir / "metrics.json"

            # Start from existing metrics if available
            existing = dict(data)
            existing.pop("config_overrides", None)
            existing.update(metrics)
            save_metrics(existing, out_metrics)

            # Update in-memory results
            all_results[name].update(metrics)
            updated_count += 1
            print(f"  OK - Updated {name}.")
        except Exception as e:
            print(f"  ERROR evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save updated kaggle_all_results.json
    if updated_count > 0:
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDone! Updated {updated_count} experiments in {all_results_path}.")
    else:
        print("\nNo experiments needed re-evaluation.")

    # Cleanup
    tmp = os.path.join(output_dir, "_re_eval.yaml")
    if os.path.isfile(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    main()
