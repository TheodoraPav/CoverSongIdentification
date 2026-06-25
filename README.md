# Cover Song Identification

Metric-learning pipeline for cover-song retrieval using frozen audio backbones (MERT, MERT-Large), a trainable MLP projection head, and systematic ablations over loss functions, segment sampling strategies, and MERT layer pooling modes.

> **Master's Thesis project** — Multimodal Machine Learning, 2nd Semester

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Configuration Reference](#configuration-reference)
- [Ablation Summary](#ablation-summary)
- [Repository Layout](#repository-layout)
- [What Is / Is Not in Git](#what-is--is-not-in-git)

---

## Overview

Given audio segments from **original** songs and their **cover** versions, we learn embeddings where covers of the same song cluster together while unrelated songs stay apart. A cover segment queries a gallery of originals; retrieval quality is measured by **MRR**, **Top-1**, **Top-5**, and **Silhouette coefficient**.

**Key design choices:**

| Component | Choice |
|-----------|--------|
| Backbone | MERT-v1-95M (frozen, 768-d) |
| Projection head | MLP with optional BatchNorm + Chroma bottleneck |
| Losses | Triplet Hard / NT-Xent / Proxy-Anchor |
| Segment sampling | Beat-aligned, dynamic pool (10 from 40) |
| Retrieval | Track-level DTW over segment sequences |

---

## Results

### Best Configuration — `best_mert_triplet_hard_feature_aug`

| Eval Level | MRR | Top-1 | Top-5 |
|---|---|---|---|
| Segment | 0.248 | 0.138 | 0.354 |
| Track Mean Pool | 0.361 | 0.250 | 0.500 |
| **Track DTW** ✓ | **0.495** | **0.417** | **0.542** |
| Track Voting | 0.463 | 0.333 | 0.542 |

> Silhouette: 0.031 · Evaluated on 480 segments (20% validation split, seeded)

### Ablation Winners

| Phase | Best Experiment | MRR |
|-------|----------------|-----|
| Layer pooling | `e1_layer_last` (layer 12) | baseline |
| Sampling / loss | `g3_beat_group_sampler_bn_triplet_hard` | best loss |
| Chroma bottleneck | `g6_chroma_48_triplet_hard` | harmonic ablation |
| Feature augmentation | `best_mert_triplet_hard_feature_aug` | **overall best** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Inference Pipeline                         │
│                                                                 │
│  audio.wav                                                      │
│     │                                                           │
│     ▼                                                           │
│  preprocess_segments.py  →  segments.csv                        │
│     │  (beat-aligned 5s clips, dynamic pool of 40)             │
│     ▼                                                           │
│  extract_features.py     →  features.pt                         │
│     │  (frozen MERT-95M, all 13 layers cached)                 │
│     ▼                                                           │
│  train.py                →  best_head.pt                        │
│     │  (MLP head: 768→512→128, BN, Dropout, L2-norm)           │
│     │  (loss: Triplet-Hard with hard negative mining)           │
│     ▼                                                           │
│  evaluate.py             →  metrics.json                        │
│     │  (DTW over segment sequences, MRR / Top-K)               │
│     ▼                                                           │
│  results/{experiment}/                                          │
│     ├── metrics.json  · history.csv  · training.log            │
│     └── curves.png · umap.png · similarity.png · silhouette.png│
└─────────────────────────────────────────────────────────────────┘
```

### Projection Head

```
Input (B, 768)
    │
    ▼
Linear(768 → 512)
    │  [optional: chroma bottleneck → Linear(768 → 48) → ReLU → Linear(48 → 512)]
    ▼
BatchNorm1d(512)  [optional]
    ▼
ReLU  →  Dropout(0.3)
    ▼
Linear(512 → 128)
    ▼
L2-normalize  →  Output (B, 128)  [unit sphere]
```

### MERT Layer Pooling Modes

| Mode | Layers | Description |
|------|--------|-------------|
| `last` | 12 | Last transformer layer (baseline) |
| `acoustic` | 0–3 | Low-level spectral / timbre |
| `early_mid` | 3–5 | Transition zone |
| `mid_beat` | 4–6 | Beat / rhythm features |
| `mid_pitch` | 6–8 | Pitch / melody / key |
| `semantic` | 9–11 | Genre / instrument |
| `musical_core` | 4–8 | Beat + pitch + harmony |
| `mean_all` | 0–12 | Equal-weight mean of all 13 layers |
| `learned_mix` | 0–12 | 13 learnable Softmax weights |

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+ with PyTorch 2.2+. GPU strongly recommended for feature extraction.

### Dataset Layout

Place `audio_manifest.csv` and `audio/*.wav` under `cover-dataset/data/`:

```
cover-dataset/data/
├── audio_manifest.csv      # group_id, role, audio_path, duration_sec, downloaded
└── audio/
    ├── <track_id>.wav
    └── ...
```

### Google Colab

1. Clone this repo and mount Google Drive.
2. Organise Drive as:
   ```
   MyDrive/CoverSongs/
   ├── cover-dataset/data/audio/       # wav files
   ├── cover-dataset/data/audio_manifest.csv
   ├── cached_features/                # created by extract_features.py
   └── checkpoints/                    # created by train.py
   ```
3. Point your YAML config (`paths.audio_root`, `cache_dir`, `checkpoints`) to Drive.
4. Run the four pipeline commands below.

### Kaggle

Use `configs/baseline_mert_ntxent_kaggle.yaml` and follow **[docs/KAGGLE.md](docs/KAGGLE.md)**.

Audio lives under `/kaggle/input/<dataset>/cover-dataset/`; outputs write to `/kaggle/working/`.

---

## Pipeline

Run all four stages with a single YAML config:

```bash
# 1. Segment audio tracks into fixed-length clips
python src/preprocess_segments.py --config configs/baseline_mert_ntxent.yaml

# 2. Extract and cache frozen MERT embeddings
python src/extract_features.py --config configs/baseline_mert_ntxent.yaml

# 3. Train the projection head
python src/train.py --config configs/baseline_mert_ntxent.yaml

# 4. Evaluate retrieval performance
python src/evaluate.py --config configs/baseline_mert_ntxent.yaml
```

To reproduce the best result:

```bash
python src/train.py    --config configs/best_mert_triplet_hard_feature_aug.yaml
python src/evaluate.py --config configs/best_mert_triplet_hard_feature_aug.yaml
```

### Output Files

| Artifact | Path | In Git? |
|----------|------|---------|
| Segment list | `data_processed/{sampling}/segments.csv` | No |
| Cached embeddings | `cached_features/{backbone}/{pool}/{augment}/{sampling}/features.pt` | No |
| Best projection head | `results/{experiment}/best_head.pt` | No |
| **Metrics (MRR, Top-K, Silhouette)** | `results/{experiment}/metrics.json` | Yes (small JSON) |
| Training history | `results/{experiment}/history.csv` | Yes |
| **Loss + metric curves** | `results/{experiment}/curves.png` | Yes |
| UMAP embedding plot | `results/{experiment}/umap.png` | Yes |
| Similarity matrix | `results/{experiment}/similarity.png` | Yes |
| Silhouette curve | `results/{experiment}/silhouette.png` | Yes |
| Training log | `results/{experiment}/training.log` | Yes |

---

## Configuration Reference

All experiment parameters live in a single YAML file. Example (`configs/baseline_mert_ntxent.yaml`):

```yaml
experiment_name: baseline_mert_ntxent
backbone: mert                          # mert | mert_large
backbone_checkpoint: m-a-p/MERT-v1-95M
sample_rate: 24000
augment: none                           # none | time
sampling: beat                          # random | stratified | beat | mixed
loss: triplet_hard                      # triplet | triplet_hard | ntxent | proxy_anchor
pool: mean                              # mean | max
segments_per_track: 10
segment_seconds: 5.0
segment_pool_mode: dynamic              # fixed | dynamic
segment_pool_max: 40
eval_level: track_dtw                   # segment | track_pool | track_dtw | track_voting
layer_pooling: last                     # see table above
seed: 42

paths:
  manifest:    cover-dataset/data/audio_manifest.csv
  cache_dir:   cached_features
  checkpoints: checkpoints
  results_dir: results

projection:
  hidden_dim: 512
  output_dim: 128
  dropout: 0.3
  batchnorm: true
  chroma_dim: 0       # 0 = disabled; 24/48/96 = harmonic bottleneck
  feature_noise: 0.02
  feature_dropout: 0.1

training:
  batch_size: 64
  epochs: 50
  lr: 1.0e-4
  weight_decay: 1.0e-4
  early_stopping_patience: 12
  use_group_batch_sampler: true
  val_fraction: 0.2
```

---

## Ablation Summary

Systematic ablations were run on Kaggle (see `notebooks/kaggle_phased_grid.py`):

| Phase | Variable | Best Setting |
|-------|----------|-------------|
| p1 | Backbone | `mert` (95M) |
| p2 | Segment sampling | `beat` |
| p3 | Loss function | `triplet_hard` |
| p4 | Layer pooling mode | `last` (layer 12) |
| p5 | Eval level | `track_dtw` |
| p6 | Segment pool | `dynamic` (10 from 40) |
| e1–e9 | MERT layer ablation | `last` |
| g3 | Group sampler + BN | `bn_triplet_hard` |
| g5 | Proxy-Anchor loss | underperforms triplet |
| g6 | Chroma bottleneck | `chroma_48` |
| best | Feature augmentation | noise + dropout ✓ |

---

## Repository Layout

```
CoverSongIdentification/
│
├── src/                        # Core ML pipeline
│   ├── preprocess_segments.py  # Stage 1: beat/random/stratified segmentation
│   ├── extract_features.py     # Stage 2: frozen MERT forward + cache
│   ├── train.py                # Stage 3: projection head training loop
│   ├── evaluate.py             # Stage 4: MRR / Top-K / Silhouette
│   ├── model.py                # BackboneSpec, LayerPooler, ProjectionHead
│   ├── dataset.py              # CachedFeatureDataset, GroupPairBatchSampler
│   ├── losses.py               # NT-Xent, Triplet (hard), Proxy-Anchor
│   ├── augmentations.py        # Waveform augmentation (pitch, stretch, gain, EQ)
│   ├── utils.py                # ExperimentConfig, path helpers, seeding, logging
│   ├── checkpointing.py        # Save / load projection head checkpoints
│   ├── proxy_bank.py           # Learnable proxy vectors (Proxy-Anchor)
│   ├── track_sequences.py      # Build ordered per-track segment sequences
│   ├── csm_matcher.py          # Stage-2 CNN on cosine-similarity matrices
│   ├── train_csm_matcher.py    # Stage-2 matcher training loop
│   ├── visualize.py            # UMAP, similarity matrix, silhouette plots
│   └── kaggle_winners.py       # Phased grid winner tracking
│
├── configs/                    # Experiment YAML files
│   ├── baseline_mert_ntxent.yaml
│   ├── best_mert_triplet_hard_feature_aug.yaml
│   └── ...
│
├── tests/                      # Unit tests
│   ├── test_group_sampler.py
│   ├── test_csm_matcher.py
│   ├── test_dynamic_segments.py
│   ├── test_early_stopping.py
│   └── test_kaggle_winners.py
│
├── notebooks/                  # Kaggle experiment runners
│   ├── kaggle_phased_grid.py   # Full phased ablation notebook
│   └── build_deck.py           # Presentation deck builder
│
├── scripts/                    # Utility scripts
│   ├── make_results_summary_table.py
│   └── repair_missing_audio.py
│
├── report/                     # LaTeX report + PDF deliverables
│   ├── main.tex / main.pdf
│   ├── ReportCoverSongIdentificationPresentation.pdf
│   ├── PresentationCoverSongIdentificationPresentation.pdf
│   └── figures/ · results/
│
├── docs/                       # Setup guides
│   ├── KAGGLE.md
│   └── GOOGLE_SLIDES.md
│
├── finalResults/               # Archived Kaggle ablation run outputs
├── cover-dataset/              # Dataset placeholder (audio not in Git)
├── requirements.txt
└── README.md
```

---

## What Is / Is Not in Git

| Tracked | Not tracked (local / Drive / Kaggle) |
|---------|--------------------------------------|
| `src/` pipeline code | `*.wav` audio files |
| `configs/` YAML files | `data_processed/` segment CSVs |
| `tests/` unit tests | `cached_features/` feature tensors |
| `results/*/metrics.json` | `checkpoints/` / `results/*/best_head.pt` |
| `results/*/curves.png` etc. | `cover-dataset/data/audio/` |
| `finalResults/` JSON + PNGs | Report + slides source files |
| `requirements.txt` | Download scripts, `.env` files |

---

## Running Tests

```bash
pytest tests/ -v
```
