# Running on Kaggle

Guide for reproducing experiments on Kaggle with GPU. Two workflows:

| Workflow | When to use |
|----------|-------------|
| **Option A** — 4-step CLI pipeline | Quick smoke test or single experiment |
| **Option B** — Phased grid notebook | Full ablation search (Phases 1–6) with winner chaining |

Base config: `configs/baseline_mert_ntxent_kaggle.yaml`  
Best result config: `configs/best_mert_triplet_hard_feature_aug.yaml`

---

## Before you start

1. **Kaggle notebook** with **GPU** enabled (Settings → Accelerator → GPU T4 x1 or better).
2. **Internet on** — MERT weights download from Hugging Face (`m-a-p/MERT-v1-95M`).
3. **Dataset** attached via **Add Data**. Expected layout:

```text
cover-dataset/
└── data/
    ├── audio/
    │   ├── <spotify_id>.wav
    │   └── ...
    └── audio_manifest.csv
```

Manifest `audio_path` values are usually `data/audio/<id>.wav`.  
Set `paths.audio_root` to the **`cover-dataset`** folder (parent of `data/`), **not** `cover-dataset/data`.

4. **Code** — clone this repo into `/kaggle/working/` (see cells below).

### Dataset mount paths

Kaggle may mount under either layout:

```text
/kaggle/input/<dataset-slug>/cover-dataset/...
/kaggle/input/datasets/<username>/<dataset-slug>/cover-dataset/...
```

Do not hard-code paths — discover them in the notebook:

```python
from pathlib import Path

manifest = next(Path("/kaggle/input").rglob("audio_manifest.csv"))
cover_root = manifest.parent.parent  # .../cover-dataset
print("manifest:", manifest)
print("audio_root:", cover_root)
print("n wav:", len(list((manifest.parent / "audio").glob("*.wav"))))
```

---

## Option A — Single experiment (CLI pipeline)

Four commands, one YAML config. Good for a quick run or reproducing the best config.

### Cell 1 — Clone + install

```python
REPO_DIR = "/kaggle/working/CoverSongIdentification"
!git clone https://github.com/TheodoraPav/CoverSongIdentification.git {REPO_DIR}
%cd {REPO_DIR}
%pip install -q -r requirements.txt
```

### Cell 2 — Patch config paths

```python
import yaml
from pathlib import Path

CONFIG = Path("configs/baseline_mert_ntxent_kaggle.yaml")
manifest = next(Path("/kaggle/input").rglob("audio_manifest.csv"))
cover_root = manifest.parent.parent

cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
cfg["paths"]["manifest"] = str(manifest)
cfg["paths"]["audio_root"] = str(cover_root)
cfg["paths"]["segments_dir"] = "/kaggle/working/data_processed"
cfg["paths"]["cache_dir"] = "/kaggle/working/cached_features"
cfg["paths"]["checkpoints"] = "/kaggle/working/checkpoints"
cfg["paths"]["results_dir"] = "/kaggle/working/results"
cfg["backbone_checkpoint"] = "m-a-p/MERT-v1-95M"
CONFIG.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
print(cfg["paths"])
```

To run the **best** configuration instead, repeat Cell 2 with  
`configs/best_mert_triplet_hard_feature_aug.yaml` (beat sampling, triplet_hard, feature aug, group sampler).

### Cell 3 — Pipeline

```python
CONFIG = "configs/baseline_mert_ntxent_kaggle.yaml"

!python src/preprocess_segments.py --config {CONFIG}
!python src/extract_features.py   --config {CONFIG} --batch-size 4
!python src/train.py              --config {CONFIG}
!python src/evaluate.py           --config {CONFIG}
```

Use `--batch-size 2` on `extract_features.py` if CUDA runs out of memory.

### Cell 4 — Visualizations (optional)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
!python src/visualize.py --config {CONFIG}
```

---

## Outputs (per experiment)

All run artifacts live under **`/kaggle/working/results/{experiment_name}/`**, not under a flat `metrics/` folder.

| Artifact | Path |
|----------|------|
| Segments | `/kaggle/working/data_processed/{sampling}/segments.csv` |
| Segments (dynamic pool) | `/kaggle/working/data_processed/{sampling}/pool_{max}/segments.csv` |
| Feature cache | `/kaggle/working/cached_features/{backbone}/{pool}/{augment}/{sampling}/features.pt` |
| Feature cache (dynamic) | `.../pool_{max}/features.pt` |
| **Checkpoint** | `/kaggle/working/results/{experiment_name}/best_head.pt` |
| **Metrics** | `/kaggle/working/results/{experiment_name}/metrics.json` |
| Training history | `/kaggle/working/results/{experiment_name}/history.csv` |
| Loss / metric curves | `/kaggle/working/results/{experiment_name}/curves.png` |
| UMAP / similarity / silhouette | `/kaggle/working/results/{experiment_name}/*.png` |
| Training log | `/kaggle/working/results/{experiment_name}/training.log` |

Example for the baseline:

```text
/kaggle/working/results/baseline_mert_ntxent/metrics.json
/kaggle/working/results/baseline_mert_ntxent/best_head.pt
```

Phased grid also writes:

| File | Purpose |
|------|---------|
| `/kaggle/working/results/kaggle_all_results.json` | Metrics + config overrides for every experiment |
| `/kaggle/working/results/kaggle_winners.json` | Phase winners (`BEST_BACKBONE`, `BEST_SAMPLING`, …) |

---

## Option B — Phased ablation grid

Full grid search across backbone, sampling, loss, augmentation, eval level, and segment pooling.  
Each experiment runs in its **own notebook cell** so Kaggle session limits are manageable.

**Source:** copy cells from `notebooks/kaggle_phased_grid.py` into a Kaggle notebook.

### How it works

1. **Cell 0** — clone repo, auto-detect dataset paths, define `make_config()` / `run_pipeline()` helpers, load prior results from disk.
2. **Cells 1–14** — one experiment per cell; winners from earlier phases are chained via `winner_overrides()`.
3. After each run — metrics saved, visualizations generated, feature cache deleted to save disk.
4. Re-run **Cell 0** after a fresh Kaggle session; later cells reload winners automatically.

Winner helpers are available in two places (equivalent API):

- Inline in `notebooks/kaggle_phased_grid.py` (Cell 0)
- Module `src/kaggle_winners.py` (for custom notebooks)

### Phase overview

| Phase | Cells | What varies | Winner key |
|-------|-------|-------------|------------|
| 1 | 1–2 | Backbone: MERT 95M vs 330M | `BEST_BACKBONE` |
| 2 | 3–6 | Sampling (stratified / beat / mixed), pool (mean / max) | `BEST_SAMPLING`, `BEST_POOL` |
| 3 | 7–8 | Loss: triplet vs triplet_hard | `BEST_LOSS` |
| 4 | 9 | Offline waveform augmentation (`time`) | `BEST_AUGMENT` |
| 5 | 10–12 | Eval level: segment / track_pool / track_dtw | `BEST_EVAL_LEVEL` |
| 6 | 13–14 | Segment pool: fixed 5 vs dynamic 5-from-20 | `BEST_SEGMENT_POOL` |

Default eval metric for winner selection: **validation MRR** (`track_dtw` in most cells).

### Winner chaining pattern

```python
from src.kaggle_winners import load_all_results, record_run, winner_overrides, set_phase_winner

ALL_RESULTS = load_all_results("/kaggle/working/results")
cfg = make_config({
    "experiment_name": "p2_sampling_beat",
    **winner_overrides("BEST_BACKBONE", results_dir="/kaggle/working/results"),
    "sampling": "beat",
})
metrics = run_training(cfg)
ALL_RESULTS = record_run("p2_sampling_beat", metrics, cfg, ALL_RESULTS, "/kaggle/working/results")
```

After the last experiment in a phase:

```python
set_phase_winner(
    "BEST_SAMPLING",
    ["p2_sampling_stratified", "p2_sampling_beat", "p2_sampling_mixed"],
    ALL_RESULTS,
    results_dir="/kaggle/working/results",
)
```

### Batch sizes (phased grid)

| Backbone | `extract_all` batch size |
|----------|--------------------------|
| MERT 95M | 8 (4 for dynamic pool) |
| MERT 330M | 2 |

### Extended ablations (after Phase 6)

Additional one-off reruns live at the repo root (copy cells or run as scripts after patching paths):

| Script | What it ablates |
|--------|-----------------|
| `kaggle_rerun_layer_ablation.py` | MERT layer pooling modes (e1–e9) |
| `kaggle_rerun_feature_aug.py` | Feature-level noise / dropout |
| `kaggle_rerun_chroma.py` | Chroma bottleneck (24 / 48 / 96) |
| `kaggle_rerun_proxy_anchor.py` | Proxy-Anchor loss |
| `kaggle_rerun_csm_matcher.py` | Stage-2 CSM + CNN matcher |
| `kaggle_best_run_loss_ablation.py` | Loss comparison on best config |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Manifest not found` | Add dataset via **Add Data**; use `rglob("audio_manifest.csv")` |
| `Skipping missing audio` / `data/data/audio` | Set `audio_root` to `cover-dataset`, not `cover-dataset/data`; re-run **preprocess** |
| `No features extracted` | Same as above; verify first segment path exists on disk |
| MERT `401` / `Repository Not Found` | Use `m-a-p/MERT-v1-95M`, not `m-audio/...` |
| CUDA OOM on extract | `--batch-size 2` (or 1 for MERT-Large) |
| MERT download fails | Enable **Internet** in notebook settings |
| `layer_pooling` requires `all_hidden` | Re-run `extract_features.py` (caches all 13 MERT layers) |
| Stale winners after config change | Delete `kaggle_winners.json` and re-run from the affected phase |

Path resolution is handled in `src/utils.py` (`resolve_audio_path`): strips duplicate folder names and normalises manifest-relative paths.

---

## Time estimates (GPU T4)

| Step | Minutes |
|------|---------|
| `preprocess_segments` | 1–5 |
| `extract_features` (MERT 95M) | 30–90 |
| `extract_features` (MERT 330M) | 60–120 |
| `train` (50 epochs, early stopping) | 10–30 |
| `evaluate` + visualizations | 3–8 |
| Full phased grid (Cells 1–14) | ~8–15 hours total |

For a quick smoke test, set `training.epochs: 5` in the YAML.

---

## Best result (reference)

Config: `configs/best_mert_triplet_hard_feature_aug.yaml`

| Setting | Value |
|---------|-------|
| Backbone | MERT 95M |
| Sampling | beat |
| Loss | triplet_hard |
| Eval level | track_dtw |
| Segment pool | dynamic (10 from 40) |
| Group batch sampler | on |
| Feature aug | noise 0.02 + dropout 0.1 |
| **Track DTW MRR** | **0.495** |

Patch Kaggle paths the same way as Option A, then run the four pipeline commands with that config file.
