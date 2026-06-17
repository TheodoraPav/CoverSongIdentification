# %% Cell 0 — Setup + Seed Previous Results
"""Kaggle multicell script for running Feature-Level Augmentation.

Goal:
  1. Seed prior results from the Kaggle dataset input (including kaggle_all_results.json)
  2. Preprocess, extract features, and run training for the feature augmentation experiment
  3. Merge results into kaggle_all_results.json
  4. Zip /kaggle/working/results → results.zip

Copy each # %% Cell N block into separate Kaggle notebook cells.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# 1. Install dependencies
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pyyaml",
        "librosa",
        "scikit-learn",
        "tqdm",
        "transformers",
        "umap-learn",
        "matplotlib",
        "audiomentations",
    ],
    check=True,
)

# 2. Clone the repository
REPO = "/kaggle/working/CoverSongIdentification"
if not os.path.isdir(REPO):
    print("Cloning repository...")
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/TheodoraPav/CoverSongIdentification.git",
            REPO,
        ],
        check=True,
    )
else:
    print("Updating repository...")
    subprocess.run(["git", "-C", REPO, "pull"], check=False)

sys.path.insert(0, REPO)
os.chdir(REPO)

import gc
import json
import torch
import yaml

from src.preprocess_segments import build_segments, write_segments
from src.extract_features import extract_all, save_features
from src.train import run_training
from src.utils import features_file_for, load_config, segments_file_for

# --- Results Persistence Utilities ---
_WINNER_KEYS = (
    "backbone", "backbone_checkpoint", "sampling", "loss", "augment", "pool",
    "eval_level", "segment_pool_mode", "segment_pool_max", "segments_per_track",
)

def _extract_overrides(cfg):
    out = {k: getattr(cfg, k) for k in _WINNER_KEYS if getattr(cfg, k, None) is not None}
    if hasattr(cfg, "projection"):
        for k in ("feature_noise", "feature_dropout", "hidden_dim", "output_dim", "dropout", "batchnorm", "chroma_dim"):
            val = getattr(cfg.projection, k, None)
            if val is not None:
                out[f"projection_{k}"] = val
    return out

def _all_results_path(results_dir):
    return os.path.join(results_dir, "kaggle_all_results.json")

def load_all_results(results_dir):
    path = _all_results_path(results_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_all_results(results, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    with open(_all_results_path(results_dir), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def record_run(name, metrics, cfg, all_results, results_dir):
    prior = dict(all_results or load_all_results(results_dir))
    n_prior = len(prior)
    store = dict(prior)
    store[name] = {**metrics, "config_overrides": _extract_overrides(cfg)}
    action = "updated" if name in prior else "added"
    print(
        f"JSON merge: {n_prior} -> {len(store)} experiments "
        f"({action} {name!r}; prior keys unchanged)"
    )
    save_all_results(store, results_dir)
    return store

def discover_previous_results_root() -> str | None:
    for candidate in (
        "/kaggle/input/datasets/theodorapavlidou/cover-song-dataset/results",
        "/kaggle/input/datasets/theodorapavlidou/cover-song-dataset/results/results",
    ):
        if os.path.isfile(os.path.join(candidate, "kaggle_all_results.json")):
            return candidate
    for hit in glob.glob("/kaggle/input/**/kaggle_all_results.json", recursive=True):
        return os.path.dirname(hit)
    return None

def seed_working_results(previous_root: str | None, working_root: str) -> int:
    os.makedirs(working_root, exist_ok=True)
    if not previous_root or not os.path.isdir(previous_root):
        print("No previous results in /kaggle/input — only new runs will be saved.")
        return 0

    n_files = 0
    for name in os.listdir(previous_root):
        src = os.path.join(previous_root, name)
        dst = os.path.join(working_root, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            n_files += sum(1 for p in Path(dst).rglob("*") if p.is_file())
        else:
            shutil.copy2(src, dst)
            n_files += 1
    print(f"Seeded {working_root} from {previous_root} (~{n_files} files)")
    return n_files

def zip_results(results_dir: str, zip_path: str) -> None:
    root = Path(results_dir)
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    n_files = sum(1 for p in root.rglob("*") if p.is_file())
    print(f"Wrote {zip_path} ({os.path.getsize(zip_path) / 2**20:.2f} MB, {n_files} files)")

# Configure paths and load config
CONFIG_PATH = "configs/best_mert_triplet_hard_feature_aug.yaml"
cfg = load_config(CONFIG_PATH)

# Set path targets to match other scripts
hits = glob.glob("/kaggle/input/**/audio_manifest.csv", recursive=True)
if not hits:
    raise FileNotFoundError(
        "audio_manifest.csv not found. Please attach the cover-song-dataset to your Kaggle notebook."
    )
MANIFEST = hits[0]
DATASET_ROOT = os.path.dirname(os.path.dirname(MANIFEST))

PATHS = {
    "manifest": MANIFEST,
    "audio_root": DATASET_ROOT,
    "segments_dir": "/kaggle/working/data_processed",
    "cache_dir": "/kaggle/working/cached_features",
    "checkpoints": "/kaggle/working/checkpoints",
    "results_dir": "/kaggle/working/results",
}
cfg.paths.manifest = PATHS["manifest"]
cfg.paths.audio_root = PATHS["audio_root"]
cfg.paths.segments_dir = PATHS["segments_dir"]
cfg.paths.cache_dir = PATHS["cache_dir"]
cfg.paths.checkpoints = PATHS["checkpoints"]
cfg.paths.results_dir = PATHS["results_dir"]

PREV_RESULTS = discover_previous_results_root()
print(f"Previous results root: {PREV_RESULTS or '(none)'}")
seed_working_results(PREV_RESULTS, PATHS["results_dir"])

ALL_RESULTS = load_all_results(PATHS["results_dir"])
print(f"Seeded {len(ALL_RESULTS)} prior experiments.")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("Ready — run Cells 1–3 in order.")


# %% Cell 1 — Preprocess + Extract Features
print("--- Preprocessing Segments ---")
write_segments(build_segments(cfg), segments_file_for(cfg))

print("\n--- Extracting MERT Features ---")
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect()
torch.cuda.empty_cache()


# %% Cell 2 — Train and Evaluate
print("\n--- Training Model with Feature Augmentation ---")
metrics = run_training(cfg)

# Save the run metrics to the main JSON tracker
RUN_NAME = cfg.experiment_name
ALL_RESULTS = record_run(RUN_NAME, metrics, cfg, ALL_RESULTS, PATHS["results_dir"])

# Try to run visualization if helper is present
try:
    print("Generating plots...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Write configuration as active for the visualizer
    with open("/kaggle/working/_active.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    subprocess.run(
        [sys.executable, "src/visualize.py", "--config", "/kaggle/working/_active.yaml"],
        check=True,
        env=env,
    )
except Exception as e:
    print(f"Visualization skipped: {e}")

gc.collect()
torch.cuda.empty_cache()


# %% Cell 3 — Verify and Export
ALL_RESULTS = load_all_results(PATHS["results_dir"])
print("\n=== Current Experiment Performance ===")
metrics = ALL_RESULTS.get(cfg.experiment_name, {})
print(f"Experiment: {cfg.experiment_name}")
print(f"MRR:        {metrics.get('mrr'):.4f}")
print(f"Top-1 Acc:  {metrics.get('top1'):.4f}")
print(f"Top-5 Acc:  {metrics.get('top5'):.4f}")
print(f"Silhouette: {metrics.get('silhouette')}")

# Add a fair comparison with previous triplet hard if available
REF_RUN = "g3_beat_group_sampler_bn_triplet_hard"
if REF_RUN in ALL_RESULTS:
    ref_metrics = ALL_RESULTS[REF_RUN]
    delta = float(metrics.get("mrr", 0.0)) - float(ref_metrics.get("mrr", 0.0))
    print(f"\n=== Comparison to Baseline ===")
    print(f"  {REF_RUN}: MRR={float(ref_metrics.get('mrr', 0.0)):.4f} (No feature aug)")
    print(f"  {cfg.experiment_name}: MRR={float(metrics.get('mrr', 0.0)):.4f} (With feature aug)")
    print(f"  Delta: {delta:+.4f}")

# Zip results to results.zip
RESULTS_ZIP = "/kaggle/working/results.zip"
zip_results(PATHS["results_dir"], RESULTS_ZIP)
print(f"\nCreated results export zip at: {RESULTS_ZIP}")
