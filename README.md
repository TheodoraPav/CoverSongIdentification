# Cover Song Identification

Metric-learning pipeline for cover-song retrieval: frozen audio backbones (MERT, MERT-Large), trainable projection head, and ablations on augmentation, loss, and segment sampling.

## What is in Git

| Included | Not in Git (local / Drive) |
|----------|----------------------------|
| `src/` pipeline code | `*.wav`, `audio_manifest.csv` |
| `configs/` experiment YAML | `data_processed/`, `cached_features/` |
| `cover-dataset/README.md`, `data/.gitkeep` | Download scripts, audio files |
| `requirements.txt` | `checkpoints/*.pt` |

Report and internal planning docs are excluded for now (see `.gitignore`).

## Pipeline

```text
preprocess_segments  →  extract_features  →  train  →  evaluate
     segments.csv         features.pt        best_head.pt   metrics JSON
```

```bash
python src/preprocess_segments.py --config configs/baseline_mert_ntxent.yaml
python src/extract_features.py   --config configs/baseline_mert_ntxent.yaml
python src/train.py              --config configs/baseline_mert_ntxent.yaml
python src/evaluate.py           --config configs/baseline_mert_ntxent.yaml
```

## Where outputs are saved

Paths come from `paths` in your YAML (`src/utils.py`).

| Artifact | Path (default) | In Git? |
|----------|----------------|---------|
| Segment list | `data_processed/{sampling}/segments.csv` | No |
| Cached embeddings | `cached_features/{backbone}/{augment}/{sampling}/features.pt` | No |
| Best projection head | `checkpoints/{backbone}/best_head.pt` | No |
| **Metrics (MRR, Top-5, Silhouette)** | `results/metrics/{experiment_name}.json` | Yes (small JSON) |
| **Figures (t-SNE / UMAP)** | `results/figures/` (planned, not generated yet) | Yes if you add PNG/PDF |

`train.py` writes metrics automatically after training. `evaluate.py` can re-run metrics from an existing checkpoint.

Example metrics file: `results/metrics/baseline_mert_ntxent.json`

```json
{
  "experiment_name": "baseline_mert_ntxent",
  "mrr": 0.42,
  "top5": 0.65,
  "silhouette": 0.12,
  "n_segments": 1200,
  "epoch": 10
}
```

## Google Colab setup

1. Clone this repo and open in Colab (or upload to Drive).
2. Mount Drive and copy data once:

   ```text
   MyDrive/CoverSongs/
   ├── cover-dataset/data/audio/          # wav files
   ├── cover-dataset/data/audio_manifest.csv
   ├── cached_features/                   # created by extract_features
   └── checkpoints/                       # created by train
   ```

3. Edit `configs/baseline_mert_ntxent.yaml` → set `paths.audio_root`, `cache_dir`, `checkpoints` to your Drive paths.
4. Install deps: `pip install -r requirements.txt`
5. Run the four commands above (same config file for all steps).

## Kaggle

Use config `configs/baseline_mert_ntxent_kaggle.yaml` and follow **[docs/KAGGLE.md](docs/KAGGLE.md)** for a full notebook example (GPU, dataset slug, four pipeline cells).

Quick path: audio under `/kaggle/input/datasets/<user>/<dataset>/cover-dataset/` (or `/kaggle/input/<dataset>/`), writes under `/kaggle/working/`. See [docs/KAGGLE.md](docs/KAGGLE.md).

## Dataset

Place `audio_manifest.csv` and `audio/*.wav` under `cover-dataset/data/` (see `cover-dataset/README.md`).  
Download scripts (`cover-dataset/src/`, `run.ps1`, root `script.py`) are **not** in Git.
