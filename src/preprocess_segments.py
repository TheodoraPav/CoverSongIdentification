"""Build `data_processed/{sampling}/segments.csv` from `audio_manifest.csv`.

For each downloaded track, emit `segments_per_track` rows with `(start_sec,
end_sec)` boundaries. Four sampling strategies:

    random      uniform random start, fixed length = `segment_seconds`.
                Starts are drawn independently, so segments may overlap.
    stratified  split the track into `n` equal zones; draw one random start
                per zone. Guarantees full coverage and no overlap (falls back
                to `random` if a zone is too short to fit `segment_seconds`).
    mixed       variable length in [segment_seconds, 2*segment_seconds].
    beat        starts snapped to beats via `librosa.beat.beat_track`;
                falls back to `random` per track if the audio is missing.

Usage:
    python src/preprocess_segments.py --config configs/<experiment>.yaml

Output columns: track_id, group_id, role, audio_path, seg_id, start_sec,
end_sec, duration_sec, sampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # support `python src/preprocess_segments.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    ExperimentConfig,
    get_logger,
    parse_config_arg,
    pool_size_for_duration,
    resolve_audio_path,
    segments_file_for,
    set_global_seed,
)

LOGGER = get_logger("preprocess_segments")


# -----------------------------------------------------------------------------
# Manifest loading
# -----------------------------------------------------------------------------


def _load_manifest(cfg: ExperimentConfig) -> pd.DataFrame:
    manifest_path = Path(cfg.paths.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Mount Drive or fix `paths.manifest` in the YAML config."
        )

    df = pd.read_csv(manifest_path)
    required = {"group_id", "role", "audio_path", "duration_sec", "downloaded"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    # CSV stores `downloaded` as text; normalize to bool.
    df["downloaded"] = df["downloaded"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    initial = len(df)
    df = df[df["downloaded"]].copy()
    LOGGER.info("Manifest: %d rows -> %d downloaded.", initial, len(df))

    # Filter out invalid, NaN, or non-positive durations before checking too_short
    invalid_dur = df["duration_sec"].isna() | (df["duration_sec"] <= 0.0)
    if invalid_dur.any():
        LOGGER.warning("Dropping %d tracks with invalid/NaN duration_sec.", int(invalid_dur.sum()))
        df = df[~invalid_dur].copy()

    too_short = df["duration_sec"] < cfg.segment_seconds
    if too_short.any():
        LOGGER.warning(
            "Dropping %d tracks shorter than %.1f sec.",
            int(too_short.sum()),
            cfg.segment_seconds,
        )
        df = df[~too_short].copy()

    df["track_id"] = df["group_id"].astype(str) + "_" + df["role"].astype(str)
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Per-track samplers
# -----------------------------------------------------------------------------


def _sample_random(
    duration: float,
    n: int,
    seg_len: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    max_start = max(0.0, duration - seg_len)
    if max_start <= 0:
        return [(0.0, float(min(duration, seg_len)))] * n
    starts = rng.uniform(0.0, max_start, size=n)
    starts.sort()
    return [(float(s), float(s + seg_len)) for s in starts]


def _sample_stratified(
    duration: float,
    n: int,
    seg_len: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Split the track into `n` equal zones and pick one random start per zone.

    Guarantees full coverage of the track and no overlap between segments.
    Falls back to `_sample_random` when the track is too short to fit `n`
    non-overlapping segments.
    """
    if duration < n * seg_len:
        return _sample_random(duration, n, seg_len, rng)

    zone_width = duration / n
    jitter_range = max(0.0, zone_width - seg_len)

    segments = []
    for i in range(n):
        zone_start = i * zone_width
        offset = rng.uniform(0.0, jitter_range) if jitter_range > 0.0 else 0.0
        start = zone_start + offset
        segments.append((float(start), float(start + seg_len)))
    return segments


def _sample_mixed(
    duration: float,
    n: int,
    seg_len: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    lengths = rng.uniform(seg_len, 2.0 * seg_len, size=n)
    segs: list[tuple[float, float]] = []
    for raw_length in lengths:
        seg_length = float(min(raw_length, duration))
        max_start = max(0.0, duration - seg_length)
        start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        segs.append((start, start + seg_length))
    segs.sort(key=lambda x: x[0])
    return segs


def _sample_beat(
    audio_path: Path,
    duration: float,
    n: int,
    seg_len: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Snap starts to detected beats; fall back to random on failure."""
    try:
        import librosa  # heavy import, only needed for beat mode

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=duration)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        max_start = max(0.0, duration - seg_len)
        valid = beat_times[beat_times <= max_start]
        if valid.size < max(2, n // 2):
            raise RuntimeError(f"too few beats: {valid.size}")

        # Pick `n` evenly spaced beats so we cover the whole track.
        indices = np.linspace(0, valid.size - 1, num=n).astype(int)
        starts = np.unique(valid[indices])
        if starts.size < n:
            extra = rng.uniform(0.0, max_start, size=n - starts.size)
            starts = np.concatenate([starts, extra])
        starts.sort()
        return [(float(s), float(s + seg_len)) for s in starts[:n]]

    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "beat sampling failed for %s (%s); using random.",
            audio_path.name,
            exc,
        )
        return _sample_random(duration, n, seg_len, rng)


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------


def build_segments(cfg: ExperimentConfig) -> pd.DataFrame:
    """Return the segments DataFrame; does not write to disk."""
    set_global_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    manifest = _load_manifest(cfg)
    seg_len = float(cfg.segment_seconds)
    rows: list[dict] = []

    for _, track in manifest.iterrows():
        duration = float(track["duration_sec"])
        manifest_audio = str(track["audio_path"]).replace("\\", "/")
        audio_path = resolve_audio_path(cfg, manifest_audio)

        if cfg.segment_pool_mode == "dynamic":
            n = pool_size_for_duration(duration, seg_len, cfg.segment_pool_max)
        else:
            n = cfg.segments_per_track

        if cfg.sampling == "random":
            segs = _sample_random(duration, n, seg_len, rng)
        elif cfg.sampling == "stratified":
            segs = _sample_stratified(duration, n, seg_len, rng)
        elif cfg.sampling == "mixed":
            segs = _sample_mixed(duration, n, seg_len, rng)
        elif cfg.sampling == "beat":
            segs = _sample_beat(audio_path, duration, n, seg_len, rng)
        else:
            raise ValueError(f"unknown sampling: {cfg.sampling}")

        for seg_id, (start, end) in enumerate(segs):
            rows.append(
                {
                    "track_id": track["track_id"],
                    "group_id": int(track["group_id"]),
                    "role": str(track["role"]),
                    # Keep manifest-relative paths so later stages re-resolve with current config.
                    "audio_path": manifest_audio,
                    "seg_id": seg_id,
                    "start_sec": round(float(start), 4),
                    "end_sec": round(float(end), 4),
                    "duration_sec": round(float(end - start), 4),
                    "sampling": cfg.sampling,
                }
            )

    if not rows:
        raise RuntimeError("No segments produced. Check the manifest filter.")

    df = pd.DataFrame(rows)
    if cfg.segment_pool_mode == "dynamic":
        avg_pool = df.groupby("track_id").size().mean()
        LOGGER.info(
            "Built %d pool segments (%d tracks, avg %.1f segs/track, "
            "dynamic pool max=%d, sampling=%s).",
            len(df),
            df["track_id"].nunique(),
            avg_pool,
            cfg.segment_pool_max,
            cfg.sampling,
        )
    else:
        LOGGER.info(
            "Built %d segments (%d tracks x %d segs/track, sampling=%s).",
            len(df),
            df["track_id"].nunique(),
            cfg.segments_per_track,
            cfg.sampling,
        )
    return df


def write_segments(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    LOGGER.info("Wrote %s (%d rows).", dest, len(df))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = parse_config_arg("Build segments.csv for the configured experiment")
    df = build_segments(cfg)
    write_segments(df, segments_file_for(cfg))


if __name__ == "__main__":
    main()
