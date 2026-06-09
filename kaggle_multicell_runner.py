# %% Cell 0 — Setup
import subprocess, sys, os, glob
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pyyaml", "librosa", "scikit-learn", "tqdm", "transformers", "umap-learn", "matplotlib", "audiomentations"], check=True)


REPO = "/kaggle/working/CoverSongIdentification"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone",
                    "https://github.com/TheodoraPav/CoverSongIdentification.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull"], check=True)
sys.path.insert(0, REPO)
os.chdir(REPO)

import gc, json, yaml, torch
from src.utils import load_config, segments_file_for, features_file_for
from src.preprocess_segments import build_segments, write_segments
from src.extract_features import extract_all, save_features
from src.train import run_training

# --- Winner persistence (inline) ---
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
    store = dict(all_results or load_all_results(results_dir))
    store[name] = {**metrics, "config_overrides": _extract_overrides(cfg)}
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
def winner_overrides(*keys, results_dir):
    winners = load_winners(results_dir)
    merged = {}
    for key in keys:
        if key not in winners:
            raise KeyError(f"Winner {key!r} not found. Available: {sorted(winners)}")
        merged.update(winners[key].get("overrides", {}))
    return merged
def print_winners(results_dir):
    winners = load_winners(results_dir)
    if not winners:
        print("(no winners saved yet)")
        return
    for key, info in winners.items():
        print(f"{key}: {info.get('experiment')}  MRR={info.get('mrr', 0):.4f}  Top1={info.get('top1', 0):.4f}")

# --- Auto-detect dataset paths ---
hits = glob.glob("/kaggle/input/**/audio_manifest.csv", recursive=True)
if not hits:
    raise FileNotFoundError("audio_manifest.csv not found under /kaggle/input/. "
                            "Add the cover-song-dataset to notebook inputs.")
MANIFEST = hits[0]
DATASET_ROOT = os.path.dirname(os.path.dirname(MANIFEST))  # up from data/ to cover-dataset/
print(f"📂 Dataset root: {DATASET_ROOT}")
print(f"📄 Manifest:     {MANIFEST}")

PATHS = {
    "manifest": MANIFEST,
    "audio_root": DATASET_ROOT,
    "segments_dir": "/kaggle/working/data_processed",
    "cache_dir": "/kaggle/working/cached_features",
    "checkpoints": "/kaggle/working/checkpoints",
    "results_dir": "/kaggle/working/results",
}

BASE_CONFIG = f"{REPO}/configs/baseline_mert_ntxent_kaggle.yaml"
ALL_RESULTS = load_all_results(PATHS["results_dir"])


def make_config(overrides, tmp="/kaggle/working/_active.yaml"):
    """Load base YAML, apply overrides + correct paths, return validated cfg."""
    with open(BASE_CONFIG) as f:
        raw = yaml.safe_load(f)
    raw.update(overrides)
    raw["paths"] = PATHS
    with open(tmp, "w") as f:
        yaml.dump(raw, f)
    return load_config(tmp)


def run_visualization(cfg):
    """Run visualize.py to generate UMAP, Similarity Comparison, and Silhouette progression plots."""
    try:
        print(f"📊 Generating visualizations (UMAP, Similarity, Silhouette) for {cfg.experiment_name}...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([sys.executable, "src/visualize.py", "--config", "/kaggle/working/_active.yaml"],
                       check=True, env=env)
        print("✅ Visualizations saved successfully!")
    except Exception as e:
        print(f"⚠️ Visualization failed for {cfg.experiment_name}: {e}")


print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print_winners(PATHS["results_dir"])
print("✅ Ready")


# %% Cell 1 — Phase 1: p1_mert (MERT 95M, ~15-25 min)
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p1_mert",
    "backbone": "mert",
    "backbone_checkpoint": "m-a-p/MERT-v1-95M",
    "sampling": "random", "loss": "triplet", "augment": "none",
    "pool": "mean", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p1_mert", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

print(f"\n✅ p1_mert: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 2 — Phase 1: p1_mert_large (MERT 330M, ~30-50 min)
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p1_mert_large",
    "backbone": "mert_large",
    "backbone_checkpoint": "m-a-p/MERT-v1-330M",
    "sampling": "random", "loss": "triplet", "augment": "none",
    "pool": "mean", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=2), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p1_mert_large", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

set_phase_winner("BEST_BACKBONE", ["p1_mert", "p1_mert_large"],
                 ALL_RESULTS, results_dir=PATHS["results_dir"])
print_winners(PATHS["results_dir"])
print(f"\n✅ p1_mert_large: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 3 — Phase 2: p2_sampling_stratified
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p2_sampling_stratified",
    **winner_overrides("BEST_BACKBONE", results_dir=PATHS["results_dir"]),
    "sampling": "stratified", "pool": "mean", "loss": "triplet",
    "augment": "none", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p2_sampling_stratified", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

print(f"\n✅ p2_sampling_stratified: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 4 — Phase 2: p2_sampling_beat
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p2_sampling_beat",
    **winner_overrides("BEST_BACKBONE", results_dir=PATHS["results_dir"]),
    "sampling": "beat", "pool": "mean", "loss": "triplet",
    "augment": "none", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p2_sampling_beat", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

print(f"\n✅ p2_sampling_beat: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 5 — Phase 2: p2_sampling_mixed
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p2_sampling_mixed",
    **winner_overrides("BEST_BACKBONE", results_dir=PATHS["results_dir"]),
    "sampling": "mixed", "pool": "mean", "loss": "triplet",
    "augment": "none", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p2_sampling_mixed", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

set_phase_winner("BEST_SAMPLING",
                 ["p2_sampling_stratified", "p2_sampling_beat", "p2_sampling_mixed"],
                 ALL_RESULTS, results_dir=PATHS["results_dir"])
print_winners(PATHS["results_dir"])
print(f"\n✅ p2_sampling_mixed: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 6 — Phase 2: p2_pool_max
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p2_pool_max",
    **winner_overrides("BEST_BACKBONE", "BEST_SAMPLING", results_dir=PATHS["results_dir"]),
    "pool": "max", "loss": "triplet", "augment": "none", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p2_pool_max", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

best_sampling = load_winners(PATHS["results_dir"])["BEST_SAMPLING"]["experiment"]
set_phase_winner("BEST_POOL", [best_sampling, "p2_pool_max"],
                 ALL_RESULTS, results_dir=PATHS["results_dir"])
print_winners(PATHS["results_dir"])
print(f"\n✅ p2_pool_max: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 7 — Phase 3: p3_ntxent
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p3_ntxent",
    **winner_overrides("BEST_BACKBONE", "BEST_SAMPLING", "BEST_POOL",
                       results_dir=PATHS["results_dir"]),
    "loss": "ntxent", "augment": "none", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p3_ntxent", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

print(f"\n✅ p3_ntxent: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 8 — Phase 3: p3_triplet_hard
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p3_triplet_hard",
    **winner_overrides("BEST_BACKBONE", "BEST_SAMPLING", "BEST_POOL",
                       results_dir=PATHS["results_dir"]),
    "loss": "triplet_hard", "augment": "none", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p3_triplet_hard", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

loss_baseline = load_winners(PATHS["results_dir"])["BEST_SAMPLING"]["experiment"]
set_phase_winner("BEST_LOSS", [loss_baseline, "p3_ntxent", "p3_triplet_hard"],
                 ALL_RESULTS, results_dir=PATHS["results_dir"])
print_winners(PATHS["results_dir"])
print(f"\n✅ p3_triplet_hard: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 9 — Phase 4: p4_time_offline
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p4_time_offline",
    **winner_overrides("BEST_BACKBONE", "BEST_SAMPLING", "BEST_POOL", "BEST_LOSS",
                       results_dir=PATHS["results_dir"]),
    "augment": "time", "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p4_time_offline", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

no_aug_baseline = load_winners(PATHS["results_dir"])["BEST_LOSS"]["experiment"]
set_phase_winner("BEST_AUGMENT", [no_aug_baseline, "p4_time_offline"],
                 ALL_RESULTS, results_dir=PATHS["results_dir"])
print_winners(PATHS["results_dir"])
print(f"\n✅ p4_time_offline: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")


# %% Cell 10 — Phase 5: p5_eval_segment
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p5_eval_segment",
    **winner_overrides("BEST_BACKBONE", "BEST_SAMPLING", "BEST_POOL",
                       "BEST_LOSS", "BEST_AUGMENT", results_dir=PATHS["results_dir"]),
    "eval_level": "segment",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p5_eval_segment", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

print(f"\n✅ p5_eval_segment: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")

# %% Cell 11 — Phase 5: p5_eval_track_pool
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p5_eval_track_pool",
    **winner_overrides("BEST_BACKBONE", "BEST_SAMPLING", "BEST_POOL",
                       "BEST_LOSS", "BEST_AUGMENT", results_dir=PATHS["results_dir"]),
    "eval_level": "track_pool",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p5_eval_track_pool", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

print(f"\n✅ p5_eval_track_pool: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")

# %% Cell 12 — Phase 5: p5_eval_track_dtw
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p5_eval_track_dtw",
    **winner_overrides("BEST_BACKBONE", "BEST_SAMPLING", "BEST_POOL",
                       "BEST_LOSS", "BEST_AUGMENT", results_dir=PATHS["results_dir"]),
    "eval_level": "track_dtw",
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p5_eval_track_dtw", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

set_phase_winner("BEST_EVAL_LEVEL",
                 ["p5_eval_segment", "p5_eval_track_pool", "p5_eval_track_dtw"],
                 ALL_RESULTS, results_dir=PATHS["results_dir"])
print_winners(PATHS["results_dir"])
print(f"\n✅ p5_eval_track_dtw: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")

# %% Cell 13 — Phase 6: p6_fixed_10 (stratified only)
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p6_fixed_10",
    **winner_overrides("BEST_BACKBONE", "BEST_POOL", "BEST_LOSS", "BEST_AUGMENT", "BEST_EVAL_LEVEL",
                       results_dir=PATHS["results_dir"]),
    "sampling": "stratified",
    "segment_pool_mode": "fixed",
    "segments_per_track": 10,
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=8), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p6_fixed_10", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

print(f"\n✅ p6_fixed_10: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")

# %% Cell 14 — Phase 6: p6_dynamic_10_40 (stratified pool)
ALL_RESULTS = load_all_results(PATHS["results_dir"])
cfg = make_config({
    "experiment_name": "p6_dynamic_10_40",
    **winner_overrides("BEST_BACKBONE", "BEST_POOL", "BEST_LOSS", "BEST_AUGMENT", "BEST_EVAL_LEVEL",
                       results_dir=PATHS["results_dir"]),
    "sampling": "stratified",
    "segment_pool_mode": "dynamic",
    "segments_per_track": 10,
    "segment_pool_max": 40,
})

write_segments(build_segments(cfg), segments_file_for(cfg))
save_features(extract_all(cfg, batch_size=4), features_file_for(cfg))
gc.collect(); torch.cuda.empty_cache()

metrics = run_training(cfg)
ALL_RESULTS = record_run("p6_dynamic_10_40", metrics, cfg, ALL_RESULTS, PATHS["results_dir"])
run_visualization(cfg)
features_file_for(cfg).unlink(missing_ok=True)
gc.collect(); torch.cuda.empty_cache()

set_phase_winner("BEST_SEGMENT_POOL", ["p6_fixed_10", "p6_dynamic_10_40"],
                 ALL_RESULTS, results_dir=PATHS["results_dir"])
print_winners(PATHS["results_dir"])
print(f"\n✅ p6_dynamic_10_40: MRR={metrics['mrr']:.4f}  Top1={metrics['top1']:.4f}")
