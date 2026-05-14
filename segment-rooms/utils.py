import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# COLMAP binary reader
# ---------------------------------------------------------------------------

def read_images_bin(path: Path):
    """Return list of dicts sorted by image name (= trajectory order).
    Each dict: {name, qvec (w,x,y,z), tvec (x,y,z)}
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack(
                "<idddddddi", f.read(64)
            )
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode())
            name = "".join(name_chars)
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2D)  # skip 2D point data
            images[image_id] = {
                "name": name,
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
            }
    return sorted(images.values(), key=lambda x: x["name"])


# ---------------------------------------------------------------------------
# Signal: DINOv2 inter-frame cosine distance
# ---------------------------------------------------------------------------

def adaptive_threshold(values: np.ndarray, window: int, sigma: float) -> np.ndarray:
    """Per-position threshold = mean(local window) + sigma * std(local window)."""
    n = len(values)
    thresholds = np.empty(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        w = values[lo:hi]
        thresholds[i] = w.mean() + sigma * (w.std() + 1e-8)
    return thresholds


def compute_dino_signal(dino_feats: np.ndarray, window: int, sigma: float):
    """
    Returns:
      delta      : float32 (N-1,)  cosine distance between consecutive frames
      threshold  : float32 (N-1,)  per-position adaptive threshold
      fires      : bool    (N-1,)  True where delta exceeds threshold
    """
    cos_sim = (dino_feats[:-1] * dino_feats[1:]).sum(axis=1).astype(np.float32)
    delta = 1.0 - cos_sim
    threshold = adaptive_threshold(delta, window, sigma)
    fires = delta > threshold
    return delta, threshold, fires


# ---------------------------------------------------------------------------
# Build segments from detected boundaries
# ---------------------------------------------------------------------------

def build_segments(n_frames: int, boundary_indices: list, frame_names: list,
                   min_len: int = 5):
    """
    Split [0, n_frames) at boundary_indices.
    Segments shorter than min_len are merged into their left neighbour.

    Returns list of dicts:
      segment_id, start_frame, end_frame, n_frames, frame_indices, frame_names
    """
    cuts = sorted(set([0] + list(boundary_indices) + [n_frames]))
    raw = [list(range(cuts[j], cuts[j + 1])) for j in range(len(cuts) - 1)]

    merged = []
    for seg in raw:
        if merged and len(seg) < min_len:
            merged[-1].extend(seg)
        else:
            merged.append(seg)

    segments = []
    for seg_id, frame_indices in enumerate(merged):
        segments.append({
            "segment_id":    seg_id,
            "start_frame":   frame_indices[0],
            "end_frame":     frame_indices[-1],
            "n_frames":      len(frame_indices),
            "frame_indices": frame_indices,
            "frame_names":   [frame_names[i] for i in frame_indices],
        })
    return segments
