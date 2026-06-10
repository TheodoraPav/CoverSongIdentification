"""Assemble per-track segment embedding sequences for track-level retrieval."""

from __future__ import annotations

import numpy as np


def build_track_sequences(data: dict) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, str]]]:
    """Group projected segment embeddings into per-track sequences sorted by seg_id.

    Returns:
        track_sequences: track_id -> (T, D) L2-normalized array
        track_info: track_id -> (group_id, role)
    """
    z = data["z"].float().cpu().numpy()
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    group_ids = np.array(data["group_id"])
    roles = np.array(data["role"])
    track_ids = np.array(data["track_id"])
    seg_ids = np.array(data["seg_id"])

    track_sequences: dict[str, np.ndarray] = {}
    track_info: dict[str, tuple[int, str]] = {}

    for tid in np.unique(track_ids):
        idx = np.where(track_ids == tid)[0]
        sorted_idx = idx[np.argsort(seg_ids[idx])]
        track_sequences[str(tid)] = z_norm[sorted_idx]
        track_info[str(tid)] = (int(group_ids[idx[0]]), str(roles[idx[0]]))

    return track_sequences, track_info
