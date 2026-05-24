# Cover song audio data (runtime)

This folder holds **audio only** for `src/preprocess_segments.py` and the rest of the ML pipeline.

## Required layout (local or Kaggle)

```text
cover-dataset/data/
├── audio/
│   └── *.wav
└── audio_manifest.csv
```

`audio_manifest.csv` must include at least: `group_id`, `role` (`original` / `cover`), `audio_path`, `duration_sec`, `downloaded`.

Point your experiment YAML to:

```yaml
paths:
  manifest: cover-dataset/data/audio_manifest.csv
  # audio_path in CSV is usually `data/audio/<id>.wav` — use cover-dataset (parent), not .../data
  audio_root: cover-dataset
```

## Not in Git

- `audio/*.wav` and `audio_manifest.csv` (size, copyright)
- Download scripts live outside this repo (see project root `.gitignore`)

Build the dataset locally, then upload to Drive / Kaggle Input as a zip.
