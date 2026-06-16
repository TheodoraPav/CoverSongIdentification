# %% Cell 0 — Setup + seed previous results
"""Kaggle multicell script: Layer-wise MERT Ablation Study.

Goal:
  1. Seed prior results from the Kaggle dataset input.
  2. Run 9 layer-pooling experiments using the locked best backbone/sampling setup:
     - e1_layer_last (last layer only - baseline)
     - e2_layer_acoustic (layers 0-3)
     - e3_layer_early_mid (layers 3-5)
     - e4_layer_mid_beat (layers 4-6)
     - e5_layer_mid_pitch (layers 6-8)
     - e6_layer_semantic (layers 9-11)
     - e7_layer_musical_core (layers 4-8)
     - e8_layer_mean_all (all 13 layers pooled with equal weights)
     - e9_layer_learned_mix (learned Softmax-blend of all 13 layers)
  3. Extract Softmax weights for the learned_mix experiment and save a bar plot.
  4. Merge new results into kaggle_all_results.json and save winners.
  5. Zip all results.

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
from src.model import build_projection_head
from src.utils import features_file_for, load_config, segments_file_for, checkpoint_path_for, pick_device

# --- Results / winners persistence (inline, same pattern as kaggle_multicell_runner.py) ---
_WINNER_KEYS = (
    "backbone", "backbone_checkpoint", "sampling", "loss", "layer_pooling", "augment", "pool",
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
            print(f"⚠️ Warning: Winner {key!r} not found in winners cache. Skipping overrides.")
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

BEST_SETUP_WINNER_KEYS = (
    "BEST_BACKBONE",
    "BEST_SAMPLING",
    "BEST_POOL",
    "BEST_AUGMENT",
    "BEST_EVAL_LEVEL",
)

SEED_MANIFEST = os.path.join(PATHS["results_dir"], "_seeded_experiment_names.json")


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
    all_results = load_all_results(results_dir)
    current_names = set(all_results)

    missing_prior = sorted(seeded_names - current_names)
    missing_new = [n for n in new_run_names if n not in current_names]
    if missing_prior:
        raise RuntimeError(
            "Prior experiments missing from kaggle_all_results.json after merge: "
            f"{missing_prior[:8]}{'...' if len(missing_prior) > 8 else ''}"
        )
    if missing_new:
        raise RuntimeError(f"New layer ablation runs missing from JSON: {missing_new}")

    stats = summarize_results_tree(results_dir)
    if seeded_names and stats["metrics"] < len(seeded_names):
        print(
            "WARNING: metrics/ file count is lower than seeded experiment count. "
            "Check that Cell 0 seeded results from /kaggle/input."
        )

    print(
        f"\nMerge OK: {len(seeded_names)} prior + {len(new_run_names)} new "
        f"= {len(current_names)} total experiments in JSON"
    )
    print_results_summary(results_dir, title="Final results folder (prior + new)")
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
    if "training" in overrides and isinstance(overrides["training"], dict):
        training = dict(raw.get("training") or {})
        training.update(overrides["training"])
        overrides = {**overrides, "training": training}
    raw.update(overrides)
    raw["paths"] = PATHS
    with open(tmp, "w", encoding="utf-8") as f:
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
        "backbone": "mert_large",
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
        "Check the dataset layout."
    )

print(f"Seeded {len(SEEDED_EXPERIMENT_NAMES)} prior experiments.")
print_results_summary(PATHS["results_dir"], title="After seed (prior only)")
print_winners(PATHS["results_dir"])
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("Ready — run Cells 1–3 in order.")


# %% Cell 1 — Preprocess + extract once (caches all_hidden tensor)
BASE_OVERRIDES = best_setup_overrides(PATHS["results_dir"])
print("Locked setup (layer pooling swept in Cell 2):")
for k, v in sorted(BASE_OVERRIDES.items()):
    print(f"  {k}: {v}")

# Extract features and cache all hidden states
cfg = make_config(
    {
        "experiment_name": "layer_ablation_shared_features",
        **BASE_OVERRIDES,
        "loss": "triplet_hard",
    }
)
write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect()
torch.cuda.empty_cache()
print(f"Features cache: {features_file_for(cfg)}")


# %% Cell 2 — Run 9 layer-wise ablation experiments
ALL_RESULTS = load_all_results(PATHS["results_dir"])
BASE_OVERRIDES = best_setup_overrides(PATHS["results_dir"])
BASE_OVERRIDES["loss"] = "triplet_hard"

LAYER_EXPERIMENTS = [
    ("e1_layer_last",         "last"),
    ("e2_layer_acoustic",     "acoustic"),
    ("e3_layer_early_mid",    "early_mid"),
    ("e4_layer_mid_beat",     "mid_beat"),
    ("e5_layer_mid_pitch",    "mid_pitch"),
    ("e6_layer_semantic",     "semantic"),
    ("e7_layer_musical_core", "musical_core"),
    ("e8_layer_mean_all",     "mean_all"),
    ("e9_layer_learned_mix",  "learned_mix"),
]

NEW_RUN_NAMES = [name for name, _ in LAYER_EXPERIMENTS]

for name, mode in LAYER_EXPERIMENTS:
    print(f"\n--- Running Experiment: {name} (layer_pooling={mode}) ---")
    cfg = make_config({
        "experiment_name": name,
        **BASE_OVERRIDES,
        "layer_pooling": mode
    })
    metrics = run_training(cfg)
    ALL_RESULTS = record_run(name, metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
    
    # Save weights plot for learned_mix
    if mode == "learned_mix":
        try:
            device = pick_device()
            head = build_projection_head(cfg).to(device)
            checkpoint_path = checkpoint_path_for(cfg)
            if checkpoint_path.is_file():
                payload = torch.load(checkpoint_path, map_location=device)
                from src.checkpointing import load_head_from_checkpoint
                load_head_from_checkpoint(head, payload, device)
                if hasattr(head, "layer_pooler") and head.layer_pooler is not None:
                    weights = head.layer_pooler.get_layer_weights()
                    print(f"Learned layer weights: {weights}")
                    
                    # Plot and save
                    import matplotlib.pyplot as plt
                    labels = ["CNN"] + [f"L{i}" for i in range(1, 13)]
                    plt.figure(figsize=(10, 5))
                    colors = ['#e74c3c' if i < 4 else '#27ae60' if i < 9 else '#3498db' for i in range(13)]
                    plt.bar(labels, weights, color=colors)
                    plt.ylabel("Softmax Weight (%)")
                    plt.title("Learned Layer Importance for Cover Song Identification")
                    plt.tight_layout()
                    fig_dir = Path(PATHS["results_dir"]) / "figures"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = fig_dir / f"{name}_layer_weights.png"
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    print(f"Saved layer weights plot to {plot_path}")
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
        except Exception as e:
            print(f"Failed to generate layer weights plot: {e}")
            
    run_visualization(cfg)
    gc.collect()
    torch.cuda.empty_cache()


# %% Cell 3 — Verify merge (prior + new ablation) + winners + zip
ALL_RESULTS = load_all_results(PATHS["results_dir"])

seeded_names = load_seed_manifest()
if not seeded_names:
    try:
        seeded_names = set(SEEDED_EXPERIMENT_NAMES)
    except NameError:
        seeded_names = {k for k in ALL_RESULTS if k not in NEW_RUN_NAMES}

verify_merged_results(PATHS["results_dir"], seeded_names, NEW_RUN_NAMES)

layer_winner = set_phase_winner(
    "BEST_LAYER_POOLING",
    NEW_RUN_NAMES,
    ALL_RESULTS,
    results_dir=PATHS["results_dir"],
    metric="mrr",
)
print(f"\nBest layer-wise pooling run: {layer_winner['experiment']}  MRR={layer_winner['mrr']:.4f}")
print_all_results(PATHS["results_dir"])
zip_results(PATHS["results_dir"], RESULTS_ZIP)
print(f"\nZip contains the FULL results/ tree: {RESULTS_ZIP}")
