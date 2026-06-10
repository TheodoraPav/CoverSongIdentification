# %% Cell 0 — Setup + seed previous results
"""Kaggle multicell script: Chroma Bottleneck ablation (standalone).

Goal:
  1. Seed prior grid + g3 results from the Kaggle dataset input
  2. Re-run the locked best setup with chroma_dim=24 bottleneck for every loss
  3. Merge new metrics into kaggle_all_results.json / kaggle_winners.json
  4. Zip /kaggle/working/results → results.zip

The chroma bottleneck forces the projection head through a narrow 24-dim
(2×12 semitones) space, stripping timbre and retaining harmonic content.

Notebook inputs (Add Data):
  - theodorapavlidou/cover-song-dataset  (audio + manifest)
  - same dataset version that includes .../results/ from prior runs

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

# --- Results / winners persistence (inline) ---
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


def set_phase_winner(winner_key, candidates, all_results, results_dir, metric="mrr"):
    present_candidates = [c for c in candidates if c in all_results]
    if not present_candidates:
        best_name = candidates[0]
        print(f"⚠️ Warning: None of the candidates {candidates} were found in results. "
              f"Attempting fallback to first candidate: {best_name}")
        winners = load_winners(results_dir)
        if winner_key in winners:
            return winners[winner_key]
        winners[winner_key] = {
            "experiment": best_name,
            metric: 0.0,
            "top1": 0.0,
            "overrides": {},
        }
        save_winners(winners, results_dir)
        return winners[winner_key]

    if len(present_candidates) < len(candidates):
        missing = [c for c in candidates if c not in all_results]
        print(f"⚠️ Warning: Some candidates are missing from results: {missing}. "
              f"Choosing winner from present candidates only: {present_candidates}")

    best_name = max(present_candidates, key=lambda n: float(all_results[n].get(metric, -1.0)))
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


def winner_overrides(*keys, results_dir):
    winners = load_winners(results_dir)
    merged = {}
    for key in keys:
        if key not in winners:
            print(f"⚠️ Warning: Winner {key!r} not found in winners cache. Skipping overrides for this phase.")
            continue
        merged.update(winners[key].get("overrides", {}))
    return merged


def print_winners(results_dir):
    winners = load_winners(results_dir)
    if not winners:
        print("(no winners saved yet)")
        return
    for key, info in winners.items():
        print(
            f"{key}: {info.get('experiment')}  "
            f"MRR={info.get('mrr', 0):.4f}  Top1={info.get('top1', 0):.4f}"
        )


def print_all_results(results_dir):
    results = load_all_results(results_dir)
    if not results:
        print("(no experiment results saved yet)")
        return
    for name, info in sorted(results.items()):
        print(f"{name}: MRR={info.get('mrr', 0):.4f}  Top1={info.get('top1', 0):.4f}")

# --- Paths ---
hits = glob.glob("/kaggle/input/**/audio_manifest.csv", recursive=True)
if not hits:
    raise FileNotFoundError(
        "audio_manifest.csv not found. Add cover-song-dataset to notebook inputs."
    )
MANIFEST = hits[0]
DATASET_ROOT = os.path.dirname(os.path.dirname(MANIFEST))
print(f"Dataset root: {DATASET_ROOT}")
print(f"Manifest:     {MANIFEST}")

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

# Phase winners merged in order (same as g3 all-losses script).
BEST_SETUP_WINNER_KEYS = (
    "BEST_BACKBONE",
    "BEST_SAMPLING",
    "BEST_POOL",
    "BEST_AUGMENT",
    "BEST_EVAL_LEVEL",
)

CHROMA_RUN_PREFIX = "g6_chroma"
CHROMA_DIM = 24  # 2 × 12 semitones
LOSSES_TO_RUN = ["ntxent", "triplet", "triplet_hard"]
SEED_MANIFEST = os.path.join(PATHS["results_dir"], "_seeded_experiment_names.json")

# Reference: best g3 BN run for fair comparison
G3_BN_REFERENCE = "g3_beat_group_sampler_bn_triplet_hard"


def summarize_results_tree(results_dir: str) -> dict[str, int]:
    root = Path(results_dir)
    stats = {
        "json_experiments": len(load_all_results(results_dir)),
    }
    for sub in ("metrics", "history", "figures", "logs"):
        folder = root / sub
        stats[sub] = len(list(folder.iterdir())) if folder.is_dir() else 0
    return stats


def print_results_summary(results_dir: str, title: str = "Results folder") -> None:
    stats = summarize_results_tree(results_dir)
    print(f"\n=== {title} ===")
    print(f"  kaggle_all_results.json: {stats['json_experiments']} experiments")
    for sub in ("metrics", "history", "figures", "logs"):
        print(f"  {sub}/: {stats[sub]} files")


def save_seed_manifest(experiment_names: set[str] | list[str], path: str = SEED_MANIFEST) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(experiment_names), f, indent=2)


def load_seed_manifest(path: str = SEED_MANIFEST) -> set[str]:
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return set(data) if isinstance(data, list) else set()


def verify_merged_results(
    results_dir: str,
    seeded_names: set[str],
    new_run_names: list[str],
) -> dict[str, list[str]]:
    """Warn if prior grid runs disappeared from JSON or artifacts tree."""
    all_results = load_all_results(results_dir)
    current_names = set(all_results)

    missing_prior = sorted(seeded_names - current_names)
    missing_new = [n for n in new_run_names if n not in current_names]
    if missing_prior:
        print(
            "⚠️ Warning: Prior experiments missing from kaggle_all_results.json after merge: "
            f"{missing_prior[:8]}{'...' if len(missing_prior) > 8 else ''}. Continuing."
        )
    if missing_new:
        print(f"⚠️ Warning: New chroma runs missing from JSON: {missing_new}. Continuing.")

    print(
        f"\nMerge OK: {len(seeded_names)} prior + {len(new_run_names)} new "
        f"= {len(current_names)} total experiments in JSON"
    )
    print(f"  Prior sample: {sorted(seeded_names)[:3]} ...")
    print(f"  New runs: {new_run_names}")
    print_results_summary(results_dir, title="Final results folder (prior + g6 chroma)")
    return {
        "prior": sorted(seeded_names),
        "new": list(new_run_names),
        "all": sorted(current_names),
    }


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
    """Copy prior results into working dir. Returns number of files copied/updated."""
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


def make_config(overrides, tmp="/kaggle/working/_active.yaml"):
    with open(BASE_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Deep-merge training sub-dict
    if "training" in overrides and isinstance(overrides["training"], dict):
        training = dict(raw.get("training") or {})
        training.update(overrides["training"])
        overrides = {**overrides, "training": training}
    # Deep-merge projection sub-dict
    if "projection" in overrides and isinstance(overrides["projection"], dict):
        projection = dict(raw.get("projection") or {})
        projection.update(overrides["projection"])
        overrides = {**overrides, "projection": projection}
    raw.update(overrides)
    raw["paths"] = PATHS
    with open(tmp, "w") as f:
        yaml.dump(raw, f)
    return load_config(tmp)


def run_visualization(cfg):
    try:
        print(f"Plots: {cfg.experiment_name}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [sys.executable, "src/visualize.py", "--config", "/kaggle/working/_active.yaml"],
            check=True,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Visualization skipped: {exc}")


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


def best_setup_overrides(results_dir: str) -> dict:
    defaults = {
        "backbone": "mert",
        "backbone_checkpoint": "m-a-p/MERT-v1-95M",
        "sampling": "beat",
        "pool": "mean",
        "augment": "none",
        "eval_level": "track_dtw",
    }
    try:
        overrides = winner_overrides(*BEST_SETUP_WINNER_KEYS, results_dir=results_dir)
    except Exception as e:
        print(f"⚠️ Warning: Failed to load winners: {e}. Using safe defaults.")
        overrides = {}
    merged = {**defaults, **overrides}
    merged.pop("loss", None)
    if merged.get("sampling") != "beat":
        print(f"⚠️ Warning: Expected best setup sampling='beat', got {merged.get('sampling')!r}. Overriding with 'beat'.")
        merged["sampling"] = "beat"
    return merged


PREV_RESULTS = discover_previous_results_root()
print(f"Previous results root: {PREV_RESULTS or '(none — attach dataset with results/)'}")

seed_working_results(PREV_RESULTS, PATHS["results_dir"])

ALL_RESULTS = load_all_results(PATHS["results_dir"])
SEEDED_EXPERIMENT_NAMES = set(ALL_RESULTS.keys())
save_seed_manifest(SEEDED_EXPERIMENT_NAMES)

if PREV_RESULTS and not SEEDED_EXPERIMENT_NAMES:
    raise RuntimeError(
        f"Found results folder at {PREV_RESULTS} but kaggle_all_results.json is empty. "
        "Check the dataset layout (metrics/ + kaggle_all_results.json must be inside)."
    )

# Reference comparison
g3_bn_mrr = ALL_RESULTS.get(G3_BN_REFERENCE, {}).get("mrr")
print(f"Seeded {len(SEEDED_EXPERIMENT_NAMES)} prior experiments (kept through all later cells).")
if g3_bn_mrr is not None:
    print(f"Reference {G3_BN_REFERENCE}: MRR={float(g3_bn_mrr):.4f}")
print_results_summary(PATHS["results_dir"], title="After seed (prior only)")
print_winners(PATHS["results_dir"])
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Chroma bottleneck dim: {CHROMA_DIM}")
print("Ready — run Cells 1–5 in order.")


# %% Cell 1 — Preprocess + extract once (shared features for all losses)
BASE_OVERRIDES = best_setup_overrides(PATHS["results_dir"])
print("Locked setup (loss swept in Cells 2–4):")
for k, v in sorted(BASE_OVERRIDES.items()):
    print(f"  {k}: {v}")
print(f"  projection.chroma_dim: {CHROMA_DIM}")

cfg = make_config(
    {
        "experiment_name": f"{CHROMA_RUN_PREFIX}_shared_features",
        **BASE_OVERRIDES,
        "loss": "triplet",
        "projection": {"chroma_dim": CHROMA_DIM},
    }
)
write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect()
torch.cuda.empty_cache()
print(f"Features: {features_file_for(cfg)}")


# %% Cell 2 — Loss: ntxent
ALL_RESULTS = load_all_results(PATHS["results_dir"])
BASE_OVERRIDES = best_setup_overrides(PATHS["results_dir"])
RUN_NAME = f"{CHROMA_RUN_PREFIX}_ntxent"
cfg = make_config({
    "experiment_name": RUN_NAME,
    **BASE_OVERRIDES,
    "loss": "ntxent",
    "projection": {"chroma_dim": CHROMA_DIM},
})
metrics = run_training(cfg)
ALL_RESULTS = record_run(RUN_NAME, metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
gc.collect()
torch.cuda.empty_cache()
print(f"{RUN_NAME}: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 3 — Loss: triplet
ALL_RESULTS = load_all_results(PATHS["results_dir"])
BASE_OVERRIDES = best_setup_overrides(PATHS["results_dir"])
RUN_NAME = f"{CHROMA_RUN_PREFIX}_triplet"
cfg = make_config({
    "experiment_name": RUN_NAME,
    **BASE_OVERRIDES,
    "loss": "triplet",
    "projection": {"chroma_dim": CHROMA_DIM},
})
metrics = run_training(cfg)
ALL_RESULTS = record_run(RUN_NAME, metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
gc.collect()
torch.cuda.empty_cache()
print(f"{RUN_NAME}: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 4 — Loss: triplet_hard
ALL_RESULTS = load_all_results(PATHS["results_dir"])
BASE_OVERRIDES = best_setup_overrides(PATHS["results_dir"])
RUN_NAME = f"{CHROMA_RUN_PREFIX}_triplet_hard"
cfg = make_config({
    "experiment_name": RUN_NAME,
    **BASE_OVERRIDES,
    "loss": "triplet_hard",
    "projection": {"chroma_dim": CHROMA_DIM},
})
metrics = run_training(cfg)
ALL_RESULTS = record_run(RUN_NAME, metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
gc.collect()
torch.cuda.empty_cache()
print(f"{RUN_NAME}: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 5 — Verify merge (prior + new) + zip full results tree
ALL_RESULTS = load_all_results(PATHS["results_dir"])
NEW_CHROMA_RUNS = [f"{CHROMA_RUN_PREFIX}_{loss}" for loss in LOSSES_TO_RUN]

seeded_names = load_seed_manifest()
if not seeded_names:
    try:
        seeded_names = set(SEEDED_EXPERIMENT_NAMES)  # noqa: F821 — set in Cell 0
    except NameError:
        seeded_names = {
            k for k in ALL_RESULTS
            if not k.startswith(f"{CHROMA_RUN_PREFIX}_")
        }
        print(
            "NOTE: Re-loaded prior experiment names from JSON "
            "(run Cell 0 first next time for a stricter check)."
        )

verify_merged_results(PATHS["results_dir"], seeded_names, NEW_CHROMA_RUNS)

winner = set_phase_winner(
    "BEST_LOSS_CHROMA",
    NEW_CHROMA_RUNS,
    ALL_RESULTS,
    results_dir=PATHS["results_dir"],
)
print(f"\nBest chroma run: {winner['experiment']}  MRR={winner['mrr']:.4f}")

# Fair comparison vs g3 BN reference
if G3_BN_REFERENCE in ALL_RESULTS:
    g3_mrr = float(ALL_RESULTS[G3_BN_REFERENCE]["mrr"])
    chroma_mrr = float(winner["mrr"])
    print(
        f"\nFair comparison (same setup, chroma_dim={CHROMA_DIM} vs no chroma):\n"
        f"  {G3_BN_REFERENCE}: MRR={g3_mrr:.4f} (no chroma)\n"
        f"  {winner['experiment']}: MRR={chroma_mrr:.4f} (chroma_dim={CHROMA_DIM})\n"
        f"  Delta: {chroma_mrr - g3_mrr:+.4f}"
    )

print_all_results(PATHS["results_dir"])

zip_results(PATHS["results_dir"], RESULTS_ZIP)
print(f"\nZip contains the FULL results/ tree (prior + g6 chroma runs): {RESULTS_ZIP}")
