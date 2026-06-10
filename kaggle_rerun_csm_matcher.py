# %% Cell 0 — Setup + seed previous results
"""Kaggle multicell script: CSM + 2D CNN matcher (modular stage 2).

Goal:
  1. Seed prior grid + g3 results from the Kaggle dataset input
  2. Reproduce the best embedding setup (g3 beat + group sampler + triplet_hard)
  3. Train CSM matcher on frozen embeddings (fair: same val split, same segments)
  4. Merge g4 CSM run into kaggle_all_results.json / kaggle_winners.json
  5. Zip /kaggle/working/results → results.zip

Fair comparison vs old runs:
  - Old runs: primary mrr = track_dtw (eval_level track_dtw)
  - New g4 run: primary mrr = track_csm_mrr (eval_level track_csm)
  - Same embedding checkpoint also reports track_dtw_mrr for DTW vs CSM on identical val

Notebook inputs (Add Data):
  - theodorapavlidou/cover-song-dataset  (audio + manifest + results/)

Prior results path (updated):
  /kaggle/input/datasets/theodorapavlidou/cover-song-dataset/results

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
from src.train_csm_matcher import run_csm_stage
from src.utils import features_file_for, load_config, segments_file_for, checkpoint_path_for

# --- Results / winners persistence (inline) ---
_WINNER_KEYS = (
    "backbone", "backbone_checkpoint", "sampling", "loss", "augment", "pool",
    "eval_level", "segment_pool_mode", "segment_pool_max", "segments_per_track",
)


def _extract_overrides(cfg):
    out = {k: getattr(cfg, k) for k in _WINNER_KEYS if getattr(cfg, k, None) is not None}
    if cfg.matcher.enabled:
        out["eval_level"] = "track_csm"
        out["matcher"] = {
            "enabled": True,
            "csm_size": cfg.matcher.csm_size,
            "csm_resize": cfg.matcher.csm_resize,
            "epochs": cfg.matcher.epochs,
            "batch_size": cfg.matcher.batch_size,
            "lr": cfg.matcher.lr,
            "negatives_per_positive": cfg.matcher.negatives_per_positive,
            "hard_negatives": cfg.matcher.hard_negatives,
            "symmetric_pairs": cfg.matcher.symmetric_pairs,
            "early_stopping_patience": cfg.matcher.early_stopping_patience,
        }
    return out


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
        mrr = info.get("mrr", 0)
        dtw = info.get("track_dtw_mrr", info.get("embedding_track_dtw_mrr"))
        extra = f"  DTW={dtw:.4f}" if dtw is not None and info.get("eval_level_config") == "track_csm" else ""
        print(f"{name}: MRR={mrr:.4f}  Top1={info.get('top1', 0):.4f}{extra}")


def format_csm_metrics_for_json(metrics: dict) -> dict:
    """Rank by CSM+CNN; keep DTW and diagonal baseline for fair comparison."""
    out = dict(metrics)
    out["experiment_name"] = metrics.get("experiment_name", CSM_RUN_NAME)
    out["mrr"] = round(float(metrics["track_csm_mrr"]), 6)
    out["top1"] = round(float(metrics["track_csm_top1"]), 6)
    out["top5"] = round(float(metrics["track_csm_top5"]), 6)
    out["eval_level_config"] = "track_csm"
    out["track_csm_diag_mrr"] = round(float(metrics.get("track_csm_diag_mrr", 0.0)), 6)
    out["track_csm_diag_top1"] = round(float(metrics.get("track_csm_diag_top1", 0.0)), 6)
    out["track_csm_diag_top5"] = round(float(metrics.get("track_csm_diag_top5", 0.0)), 6)
    out["embedding_track_dtw_mrr"] = round(float(metrics.get("track_dtw_mrr", 0.0)), 6)
    out["embedding_track_dtw_top1"] = round(float(metrics.get("track_dtw_top1", 0.0)), 6)
    out["embedding_track_dtw_top5"] = round(float(metrics.get("track_dtw_top5", 0.0)), 6)
    return out


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

# Experiment names
EMBED_RUN_NAME = "g3_beat_group_sampler_triplet_hard"
CSM_RUN_NAME = "g4_csm_beat_group_sampler_triplet_hard"
NEW_RUN_NAMES = [CSM_RUN_NAME]

# Default matcher hyperparams (stage 2 only)
MATCHER_OVERRIDES = {
    "enabled": True,
    "csm_size": 10,
    "csm_resize": False,
    "epochs": 30,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "negatives_per_positive": 4,
    "hard_negatives": True,
    "symmetric_pairs": True,
    "early_stopping_patience": 5,
}

SEED_MANIFEST = os.path.join(PATHS["results_dir"], "_seeded_experiment_names.json")


def summarize_results_tree(results_dir: str) -> dict[str, int]:
    root = Path(results_dir)
    stats = {"json_experiments": len(load_all_results(results_dir))}
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
        raise RuntimeError(f"New CSM run missing from JSON: {missing_new}")

    print(
        f"\nMerge OK: {len(seeded_names)} prior + {len(new_run_names)} new "
        f"= {len(current_names)} total experiments in JSON"
    )
    print(f"  Prior sample: {sorted(seeded_names)[:3]} ...")
    print(f"  New runs: {new_run_names}")
    print_results_summary(results_dir, title="Final results folder (prior + g4 CSM)")
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
    if "matcher" in overrides and isinstance(overrides["matcher"], dict):
        matcher = dict(raw.get("matcher") or {})
        matcher.update(overrides["matcher"])
        overrides = {**overrides, "matcher": matcher}
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


def best_embed_overrides(results_dir: str) -> dict:
    """Locked g3 setup: beat + group sampler + triplet_hard + track_dtw eval."""
    winners = load_winners(results_dir)
    key = "BEST_LOSS_BEAT_GROUP_SAMPLER"
    if key not in winners:
        raise KeyError(
            f"{key!r} not found in kaggle_winners.json. "
            "Attach a results/ folder that includes the g3 group-sampler runs."
        )
    overrides = dict(winners[key].get("overrides", {}))
    if overrides.get("sampling") != "beat":
        raise RuntimeError(f"Expected sampling='beat', got {overrides.get('sampling')!r}")
    if overrides.get("loss") != "triplet_hard":
        raise RuntimeError(f"Expected loss='triplet_hard', got {overrides.get('loss')!r}")
    overrides.setdefault("eval_level", "track_dtw")
    return overrides


PREV_RESULTS = discover_previous_results_root()
print(f"Previous results root: {PREV_RESULTS or '(none — attach dataset with results/)'}")

seed_working_results(PREV_RESULTS, PATHS["results_dir"])

ALL_RESULTS = load_all_results(PATHS["results_dir"])
SEEDED_EXPERIMENT_NAMES = set(ALL_RESULTS.keys())
save_seed_manifest(SEEDED_EXPERIMENT_NAMES)

if PREV_RESULTS and not SEEDED_EXPERIMENT_NAMES:
    raise RuntimeError(
        f"Found results folder at {PREV_RESULTS} but kaggle_all_results.json is empty."
    )

g3_mrr = ALL_RESULTS.get(EMBED_RUN_NAME, {}).get("mrr")
print(f"Seeded {len(SEEDED_EXPERIMENT_NAMES)} prior experiments.")
if g3_mrr is not None:
    print(f"Prior {EMBED_RUN_NAME}: DTW MRR={float(g3_mrr):.4f} (comparison baseline)")
print_results_summary(PATHS["results_dir"], title="After seed (prior only)")
print_winners(PATHS["results_dir"])
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("Ready — run Cells 1–4 in order.")


# %% Cell 1 — Preprocess + extract once (same features as g3)
BASE_OVERRIDES = best_embed_overrides(PATHS["results_dir"])
print("Locked embedding setup (stage 1):")
for k, v in sorted(BASE_OVERRIDES.items()):
    print(f"  {k}: {v}")

cfg = make_config(
    {
        "experiment_name": f"{CSM_RUN_NAME}_shared_features",
        **BASE_OVERRIDES,
    }
)
write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect()
torch.cuda.empty_cache()
print(f"Features: {features_file_for(cfg)}")


# %% Cell 2 — Stage 1: train projection head (skip if checkpoint already exists)
BASE_OVERRIDES = best_embed_overrides(PATHS["results_dir"])
cfg = make_config(
    {
        "experiment_name": EMBED_RUN_NAME,
        **BASE_OVERRIDES,
        "matcher": {"enabled": False},
    }
)
ckpt = checkpoint_path_for(cfg)
if ckpt.is_file():
    print(f"Skipping embed training — checkpoint exists: {ckpt}")
    if EMBED_RUN_NAME in load_all_results(PATHS["results_dir"]):
        prior = load_all_results(PATHS["results_dir"])[EMBED_RUN_NAME]
        print(
            f"  Seeded {EMBED_RUN_NAME}: DTW MRR={prior.get('mrr', 0):.4f} "
            "(from input results; not overwritten)"
        )
else:
    print(f"Training projection head: {EMBED_RUN_NAME}")
    embed_metrics = run_training(cfg)
    run_visualization(cfg)
    print(
        f"{EMBED_RUN_NAME}: DTW MRR={embed_metrics['mrr']:.4f}  "
        f"Top1={embed_metrics['top1']:.4f}"
    )
    seeded = load_all_results(PATHS["results_dir"])
    if EMBED_RUN_NAME in seeded:
        print(
            f"NOTE: not overwriting seeded JSON entry for {EMBED_RUN_NAME!r} "
            f"(input had MRR={seeded[EMBED_RUN_NAME].get('mrr', 0):.4f})"
        )

gc.collect()
torch.cuda.empty_cache()


# %% Cell 3 — Stage 2: CSM + CNN matcher on frozen embeddings
ALL_RESULTS = load_all_results(PATHS["results_dir"])
BASE_OVERRIDES = best_embed_overrides(PATHS["results_dir"])

cfg = make_config(
    {
        "experiment_name": CSM_RUN_NAME,
        **BASE_OVERRIDES,
        "matcher": MATCHER_OVERRIDES,
    }
)
if not cfg.matcher.enabled:
    raise RuntimeError("matcher.enabled must be true for CSM stage")

raw_metrics = run_csm_stage(cfg)
metrics = format_csm_metrics_for_json(raw_metrics)
ALL_RESULTS = record_run(CSM_RUN_NAME, metrics, cfg, ALL_RESULTS, PATHS["results_dir"])

gc.collect()
torch.cuda.empty_cache()
print(
    f"\n{CSM_RUN_NAME}:\n"
    f"  CSM+CNN  MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}\n"
    f"  CSM diag MRR={metrics.get('track_csm_diag_mrr', 0):.4f}  (no CNN baseline)\n"
    f"  DTW      MRR={metrics['embedding_track_dtw_mrr']:.4f}  (same embeddings)"
)
if g3_mrr := ALL_RESULTS.get(EMBED_RUN_NAME, {}).get("mrr"):
    delta = metrics["mrr"] - float(g3_mrr)
    print(f"  vs seeded {EMBED_RUN_NAME} DTW MRR={float(g3_mrr):.4f}  "
          f"(CSM−DTW on new run: {delta:+.4f})")


# %% Cell 4 — Verify merge (prior + g4) + winners + zip
ALL_RESULTS = load_all_results(PATHS["results_dir"])

seeded_names = load_seed_manifest()
if not seeded_names:
    try:
        seeded_names = set(SEEDED_EXPERIMENT_NAMES)  # noqa: F821
    except NameError:
        seeded_names = {k for k in ALL_RESULTS if k != CSM_RUN_NAME}
        print("NOTE: re-loaded prior names from JSON (run Cell 0 first for stricter check).")

verify_merged_results(PATHS["results_dir"], seeded_names, NEW_RUN_NAMES)

csm_winner = set_phase_winner(
    "BEST_CSM_MATCHER",
    NEW_RUN_NAMES,
    ALL_RESULTS,
    results_dir=PATHS["results_dir"],
    metric="mrr",
)
print(f"\nBest CSM run: {csm_winner['experiment']}  CSM MRR={csm_winner['mrr']:.4f}")

if EMBED_RUN_NAME in ALL_RESULTS:
    dtw_baseline = float(ALL_RESULTS[EMBED_RUN_NAME]["mrr"])
    csm_mrr = float(ALL_RESULTS[CSM_RUN_NAME]["mrr"])
    csm_diag = float(ALL_RESULTS[CSM_RUN_NAME].get("track_csm_diag_mrr", 0))
    print(
        f"\nFair comparison (same val protocol, same g3 hyperparams):\n"
        f"  {EMBED_RUN_NAME}       track_dtw      MRR={dtw_baseline:.4f}\n"
        f"  {CSM_RUN_NAME}  track_csm_diag MRR={csm_diag:.4f}\n"
        f"  {CSM_RUN_NAME}  track_csm+CNN  MRR={csm_mrr:.4f}  "
        f"(Δ vs DTW={csm_mrr - dtw_baseline:+.4f})"
    )

print_all_results(PATHS["results_dir"])
zip_results(PATHS["results_dir"], RESULTS_ZIP)
print(f"\nZip contains FULL results/ tree (prior grid + g3 + g4 CSM): {RESULTS_ZIP}")
