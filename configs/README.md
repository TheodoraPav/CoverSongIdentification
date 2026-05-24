# Experiment configs

One YAML per run. All pipeline scripts take the same file:

```bash
python src/<script>.py --config configs/<name>.yaml
```

## Baseline

| File | Purpose |
|------|---------|
| `baseline_mert_ntxent.yaml` | MERT 95M, local paths |
| `baseline_mert_ntxent_kaggle.yaml` | Same, paths for Kaggle (`/kaggle/input` + `/kaggle/working`) |
| `baseline_mert_ntxent_colab.yaml` | Same, Google Drive paths |

## Colab / Kaggle paths

Override in YAML (example):

```yaml
paths:
  manifest: /content/drive/MyDrive/CoverSongs/cover-dataset/data/audio_manifest.csv
  # Manifest paths are `data/audio/*.wav` → audio_root is cover-dataset (parent of data/)
  audio_root: /content/drive/MyDrive/CoverSongs/cover-dataset
  cache_dir: /content/drive/MyDrive/CoverSongs/cached_features
  checkpoints: /content/drive/MyDrive/CoverSongs/checkpoints
  results_dir: results
```

`results_dir` can stay `results` inside the repo clone (metrics JSON is small).

## Output paths (auto from config fields)

| Stage | Path pattern |
|-------|----------------|
| Segments | `{segments_dir}/{sampling}/segments.csv` |
| Features | `{cache_dir}/{backbone}/{augment}/{sampling}/features.pt` |
| Checkpoint | `{checkpoints}/{backbone}/best_head.pt` |
| Metrics | `{results_dir}/metrics/{experiment_name}.json` |

Examples:

```text
data_processed/random/segments.csv
data_processed/beat/segments.csv
cached_features/mert/none/random/features.pt
results/metrics/baseline_mert_ntxent.json
```

Change only `sampling` in YAML to get a new segments file; change `experiment_name` for a new metrics JSON.
