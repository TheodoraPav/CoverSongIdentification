# %% Cell 0 — Setup + seed previous results
"""Kaggle multicell script: Proxy-Anchor loss (single fair ablation vs g3).

Goal:
  1. Seed prior grid + g3 + g4 results from the Kaggle dataset input
  2. Re-run the locked g3 setup with loss=proxy_anchor only
  3. Merge g5 into kaggle_all_results.json / kaggle_winners.json
  4. Zip /kaggle/working/results → results.zip

Fair comparison:
  g3_beat_group_sampler_triplet_hard  →  MRR 0.538 (DTW)
  g5_beat_group_sampler_proxy_anchor  →  MRR ? (same beat / group sampler / eval)

Notebook inputs (Add Data):
  - theodorapavlidou/cover-song-dataset  (audio + manifest + latest results/)

Copy each  # %% Cell N  block into a separate Kaggle notebook cell.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

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

REPO = "/kaggle/working/CoverSongIdentification"
if not os.path.isdir(REPO):
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
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)

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

_WINNER_KEYS = (
    "backbone", "backbone_checkpoint", "sampling", "loss", "augment", "pool",
    "eval_level", "segment_pool_mode", "segment_pool_max", "segments_per_track",
)


def _extract_overrides(cfg):
    return {k: getattr(cfg, k) for k in _WINNER_KEYS if getattr(cfg, k, None) is not None}


def _all_results_path(results_dir):
    return os.path.join(results_dir, "kaggle_all_results.json")


def _winners_path(results_dir):
    return os.path.join(results_dir, "kaggle_winners.json")


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


def load_winners(results_dir):
    path = _winners_path(results_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_winners(winners, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    with open(_winners_path(results_dir), "w", encoding="utf-8") as f:
        json.dump(winners, f, indent=2)


def record_run(name, metrics, cfg, all_results, results_dir):
    prior = dict(all_results or load_all_results(results_dir))
    store = dict(prior)
    store[name] = {**metrics, "config_overrides": _extract_overrides(cfg)}
    print(f"JSON merge: {len(prior)} -> {len(store)} experiments (added/updated {name!r})")
    save_all_results(store, results_dir)
    return store


def set_phase_winner(winner_key, candidates, all_results, results_dir, metric="mrr"):
    missing = [c for c in candidates if c not in all_results]
    if missing:
        raise KeyError(f"Missing results for: {missing}")
    best_name = max(candidates, key=lambda n: float(all_results[n].get(metric, -1.0)))
    payload = all_results[best_name]
    winners = load_winners(results_dir)
    winners[winner_key] = {
        "experiment": best_name,
        metric: float(payload.get(metric, 0.0)),
        "top1": float(payload.get("top1", 0.0)),
        "overrides": payload.get("config_overrides", {}),
    }
    save_winners(winners, results_dir)
    return winners[winner_key]


def print_all_results(results_dir):
    results = load_all_results(results_dir)
    for name, info in sorted(results.items()):
        print(f"{name}: MRR={info.get('mrr', 0):.4f}  Top1={info.get('top1', 0):.4f}")


hits = glob.glob("/kaggle/input/**/audio_manifest.csv", recursive=True)
if not hits:
    raise FileNotFoundError("audio_manifest.csv not found.")
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

BASE_CONFIG = f"{REPO}/configs/baseline_mert_ntxent_kaggle.yaml"
RESULTS_ZIP = "/kaggle/working/results.zip"
RUN_NAME = "g5_beat_group_sampler_proxy_anchor"
G3_REFERENCE = "g3_beat_group_sampler_triplet_hard"
SEED_MANIFEST = os.path.join(PATHS["results_dir"], "_seeded_experiment_names.json")


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


def seed_working_results(previous_root: str | None, working_root: str) -> None:
    os.makedirs(working_root, exist_ok=True)
    if not previous_root or not os.path.isdir(previous_root):
        print("No previous results in /kaggle/input")
        return
    for name in os.listdir(previous_root):
        src = os.path.join(previous_root, name)
        dst = os.path.join(working_root, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"Seeded {working_root} from {previous_root}")


def make_config(overrides, tmp="/kaggle/working/_active.yaml"):
    with open(BASE_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "training" in overrides and isinstance(overrides["training"], dict):
        training = dict(raw.get("training") or {})
        training.update(overrides["training"])
        overrides = {**overrides, "training": training}
    raw.update(overrides)
    raw["paths"] = PATHS
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(raw, f)
    return load_config(tmp)


def best_g3_setup_overrides(results_dir: str) -> dict:
    winners = load_winners(results_dir)
    key = "BEST_LOSS_BEAT_GROUP_SAMPLER"
    if key not in winners:
        raise KeyError(f"{key!r} missing from kaggle_winners.json")
    overrides = dict(winners[key].get("overrides", {}))
    overrides.pop("loss", None)
    if overrides.get("sampling") != "beat":
        raise RuntimeError(f"Expected beat sampling, got {overrides.get('sampling')!r}")
    return overrides


def zip_results(results_dir: str, zip_path: str) -> None:
    root = Path(results_dir)
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    print(f"Wrote {zip_path}")


PREV_RESULTS = discover_previous_results_root()
seed_working_results(PREV_RESULTS, PATHS["results_dir"])
ALL_RESULTS = load_all_results(PATHS["results_dir"])
with open(SEED_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(sorted(ALL_RESULTS.keys()), f, indent=2)

g3_mrr = ALL_RESULTS.get(G3_REFERENCE, {}).get("mrr")
print(f"Seeded {len(ALL_RESULTS)} experiments. {G3_REFERENCE} MRR={g3_mrr}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# %% Cell 1 — Preprocess + extract (same features as g3)
BASE_OVERRIDES = best_g3_setup_overrides(PATHS["results_dir"])
cfg = make_config(
    {
        "experiment_name": f"{RUN_NAME}_shared_features",
        **BASE_OVERRIDES,
        "loss": "proxy_anchor",
    }
)
write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect()
torch.cuda.empty_cache()


# %% Cell 2 — Train Proxy-Anchor (g5)
ALL_RESULTS = load_all_results(PATHS["results_dir"])
BASE_OVERRIDES = best_g3_setup_overrides(PATHS["results_dir"])
cfg = make_config(
    {
        "experiment_name": RUN_NAME,
        **BASE_OVERRIDES,
        "loss": "proxy_anchor",
        "training": {
            "proxy_alpha": 32,
            "proxy_delta": 0.1,
        },
    }
)
metrics = run_training(cfg)
ALL_RESULTS = record_run(RUN_NAME, metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
gc.collect()
torch.cuda.empty_cache()
print(f"{RUN_NAME}: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")
if g3_mrr is not None:
    print(f"vs {G3_REFERENCE}: MRR={float(g3_mrr):.4f}  (delta={metrics['mrr'] - float(g3_mrr):+.4f})")


# %% Cell 3 — Verify merge + winner + zip
ALL_RESULTS = load_all_results(PATHS["results_dir"])
with open(SEED_MANIFEST, encoding="utf-8") as f:
    seeded = set(json.load(f))
missing_prior = sorted(seeded - set(ALL_RESULTS))
if missing_prior:
    raise RuntimeError(f"Prior experiments missing after merge: {missing_prior[:5]}")
if RUN_NAME not in ALL_RESULTS:
    raise RuntimeError(f"Missing new run {RUN_NAME!r}")

winner = set_phase_winner(
    "BEST_PROXY_ANCHOR_BEAT_GROUP_SAMPLER",
    [RUN_NAME],
    ALL_RESULTS,
    results_dir=PATHS["results_dir"],
)
print(f"Winner: {winner['experiment']}  MRR={winner['mrr']:.4f}")
if G3_REFERENCE in ALL_RESULTS:
    g3 = float(ALL_RESULTS[G3_REFERENCE]["mrr"])
    g5 = float(ALL_RESULTS[RUN_NAME]["mrr"])
    print(f"\nFair loss ablation (same setup, track_dtw eval):")
    print(f"  {G3_REFERENCE}: triplet_hard  MRR={g3:.4f}")
    print(f"  {RUN_NAME}: proxy_anchor MRR={g5:.4f}  (delta={g5 - g3:+.4f})")
print_all_results(PATHS["results_dir"])
zip_results(PATHS["results_dir"], RESULTS_ZIP)
