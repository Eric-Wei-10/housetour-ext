"""
segment_rooms.py  —  Room-based Segmentation

Classifies each keyframe with SigLIP and segments the sequence by detecting
where the predicted room label changes between consecutive frames.

Output per scene (<output_root>/<scene_id>/):
  siglip_features.npy    float32 (N, D)   L2-normalised SigLIP embeddings
  room_labels.npy        int32   (N,)     per-frame room label index
  room_scores.npy        float32 (N, C)   per-frame per-category scores
  segments.json          list of segment dicts
  boundaries.txt         detected boundary frame indices
  segment_grid.png       thumbnail grid — representative frames per segment
  frames/seg_xx/         keyframe copies with top-2 room labels overlaid

Usage:
  python ./segment-rooms/segment_rooms.py \
      --data_root /cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions \
      --output_root /cluster/project/cvg/students/xinwei/official-housetour-dataset/label_segments \
      --batch_size 32 \
      --scene_ids 7
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from siglip import (load_siglip_model, extract_siglip_features,
                    build_room_text_embeddings, classify_rooms,
                    HOUSE_ROOM_TYPES)
from utils import read_images_bin
from visualization import (save_segment_grid, save_similarity_matrix,
                            save_segment_frames)


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

WINDOW_HALF     = 5     # w: voting window covers [n-w, n+w]
MIN_SEGMENT_LEN = 5
SCORE_THRESHOLD = 0.01  # drop segments whose top-1 score never exceeds this
DROP_LABELS     = {"hallway"}  # drop segments whose majority label is one of these
THUMB_COLS      = 5


# ---------------------------------------------------------------------------
# Label-based boundary detection
# ---------------------------------------------------------------------------

def sliding_window_vote(scores: np.ndarray, w: int) -> np.ndarray:
    """Smooth per-frame labels using a sliding window vote over top-2 labels.

    For each frame n the window [n-w, n+w] is collected. Each frame contributes
    2 votes (its top-2 labels). Ties in vote count are broken by the sum of the
    corresponding per-frame scores accumulated across the window.

    Args:
        scores: float32 (N, C)  per-frame per-category scores
        w:      half-window size (full window = 2w+1)
    Returns:
        smoothed_labels: int32 (N,)
    """
    n, c = scores.shape
    # argsort ascending: col 0 = 2nd-best (top-2), col 1 = best (top-1)
    top2 = np.argsort(scores, axis=1)[:, -2:]   # (N, 2)
    smoothed = np.empty(n, dtype=np.int32)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        frame_range  = np.arange(lo, hi)
        top1_labels  = top2[lo:hi, 1]   # best label per frame  → 2 votes
        top2_labels  = top2[lo:hi, 0]   # 2nd-best label per frame → 1 vote
        vote_counts  = (np.bincount(top1_labels, minlength=c) * 2 +
                        np.bincount(top2_labels, minlength=c) * 1)

        # Tiebreaker: sum of scores from frames that voted for each label
        score_sums = np.zeros(c, dtype=np.float64)
        np.add.at(score_sums, top1_labels, scores[frame_range, top1_labels])
        np.add.at(score_sums, top2_labels, scores[frame_range, top2_labels])

        max_votes = vote_counts.max()
        candidates = np.where(vote_counts == max_votes)[0]
        smoothed[i] = candidates[score_sums[candidates].argmax()]
    return smoothed


def trim_segment(seg: dict, room_labels: np.ndarray,
                 smoothed_labels: np.ndarray, min_len: int,
                 frames_root: Path) -> dict | None:
    """Trim a segment to the span where raw top-1 == smoothed label.

    Scans from front and back for the first frame whose raw top-1 label matches
    the segment's smoothed label.  All files outside that span are deleted from
    frames_root/seg_XX/.  Returns the trimmed segment dict, or None if dropped.
    """
    indices   = seg["frame_indices"]
    names     = seg["frame_names"]
    seg_label = int(smoothed_labels[indices[0]])
    frame_dir = frames_root / f"seg_{seg['segment_id']:02d}"

    front = next((k for k, fi in enumerate(indices)
                  if room_labels[fi] == seg_label), None)
    back  = next((k for k in range(len(indices) - 1, -1, -1)
                  if room_labels[indices[k]] == seg_label), None)

    trimmed_names = [] if front is None else names[front : back + 1]

    # Delete every file in frame_dir not in the kept set
    kept_set = set(trimmed_names)
    if frame_dir.exists():
        for p in frame_dir.iterdir():
            if p.name not in kept_set:
                p.unlink()
                print(f"    deleted {frame_dir.name}/{p.name}")

    if front is None:
        print(f"    drop seg {seg['segment_id']} — no frame matches smoothed label")
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        return None

    trimmed_indices = indices[front : back + 1]

    if len(trimmed_indices) < min_len:
        print(f"    drop seg {seg['segment_id']} — only {len(trimmed_indices)} frames after trim (< {min_len})")
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        return None

    return {
        "segment_id":    seg["segment_id"],
        "start_frame":   trimmed_indices[0],
        "end_frame":     trimmed_indices[-1],
        "n_frames":      len(trimmed_indices),
        "frame_indices": trimmed_indices,
        "frame_names":   trimmed_names,
    }


def find_label_boundaries(labels: np.ndarray) -> list[int]:
    """Return frame indices where the room label changes from the previous frame."""
    changes = np.where(labels[1:] != labels[:-1])[0] + 1
    return changes.tolist()


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_scene(
    scene_dir:       Path,
    output_dir:      Path,
    siglip_model,
    device:          str,
    batch_size:      int,
    window_half:     int,
    min_len:         int,
    score_threshold: float,
):
    images_bin    = scene_dir / "0_metric" / "images.bin"
    keyframes_dir = scene_dir / "keyframes"

    if not images_bin.exists():
        print(f"  [skip] no 0_metric/images.bin in {scene_dir.name}")
        return
    if not keyframes_dir.exists():
        print(f"  [skip] no keyframes/ in {scene_dir.name}")
        return

    if output_dir.exists():
        print(f"  [skip] output_dir already exists: {output_dir}")
        return

    output_dir.mkdir(parents=True)

    entries = read_images_bin(images_bin)

    image_paths, frame_names = [], []
    missing = 0
    for entry in entries:
        img_path = keyframes_dir / entry["name"]
        if not img_path.exists():
            missing += 1
            continue
        image_paths.append(img_path)
        frame_names.append(entry["name"])

    if missing:
        print(f"  [warn] {missing} frames listed in images.bin but missing from keyframes/")
    if len(image_paths) < 2:
        print(f"  [skip] too few valid frames ({len(image_paths)}) in {scene_dir.name}")
        return

    n = len(image_paths)

    # ── SigLIP features + room classification ─────────────────────────────
    siglip_feats = extract_siglip_features(image_paths, siglip_model, device, batch_size)
    np.save(output_dir / "siglip_features.npy", siglip_feats)

    text_embeddings, logit_scale, logit_bias = build_room_text_embeddings(siglip_model, device)
    room_labels, room_scores = classify_rooms(siglip_feats, text_embeddings, logit_scale, logit_bias)
    np.save(output_dir / "room_labels.npy", room_labels)
    np.save(output_dir / "room_scores.npy", room_scores)

    # ── Sliding-window vote to smooth labels ──────────────────────────────
    smoothed_labels = sliding_window_vote(room_scores, window_half)
    np.save(output_dir / "room_labels_smoothed.npy", smoothed_labels)

    # ── Segmentation by label change ──────────────────────────────────────
    boundary_indices = find_label_boundaries(smoothed_labels)
    boundary_info = [(i, frame_names[i]) for i in boundary_indices]
    print(f"Boundary indices: {boundary_info}")

    cuts = sorted({0} | set(boundary_indices) | {n})
    segments = []
    for seg_id, (start, end) in enumerate(zip(cuts, cuts[1:])):
        frame_indices = list(range(start, end))
        segments.append({
            "segment_id":    seg_id,
            "start_frame":   start,
            "end_frame":     end - 1,
            "n_frames":      len(frame_indices),
            "frame_indices": frame_indices,
            "frame_names":   [frame_names[i] for i in frame_indices],
        })

    # ── Save data ─────────────────────────────────────────────────────────
    with open(output_dir / "segments.json", "w") as f:
        json.dump(segments, f, indent=2)
    (output_dir / "boundaries.txt").write_text(
        "\n".join(map(str, boundary_indices)) + "\n")

    # ── Visualize ─────────────────────────────────────────────────────────
    save_segment_grid(segments, image_paths, output_dir / "segment_grid.png",
                      cols=THUMB_COLS)
    save_similarity_matrix(siglip_feats, output_dir / "similarity_matrix_siglip.png")
    save_similarity_matrix(siglip_feats, output_dir / "similarity_matrix_siglip_segmented.png",
                           segments=segments)
    save_segment_frames(segments, image_paths, output_dir,
                        room_labels=room_labels, room_scores=room_scores,
                        category_names=HOUSE_ROOM_TYPES,
                        smoothed_labels=smoothed_labels)

    # ── Trim & filter short segments ──────────────────────────────────────
    frames_root = output_dir / "frames"
    kept = []
    for seg in segments:
        result = trim_segment(seg, room_labels, smoothed_labels, min_len, frames_root)
        if result is not None:
            kept.append(result)
    # Keep original segment_id so it matches the frames/seg_XX folder name.
    n_after_trim = len(kept)

    # ── Drop low-confidence segments ──────────────────────────────────────
    _drop_label_ids = {HOUSE_ROOM_TYPES.index(l) for l in DROP_LABELS
                       if l in HOUSE_ROOM_TYPES}
    final = []
    for seg in kept:
        top1_max = room_scores[seg["frame_indices"]].max(axis=1).max()
        seg_label = int(smoothed_labels[seg["frame_indices"][0]])
        frame_dir = frames_root / f"seg_{seg['segment_id']:02d}"
        if top1_max < score_threshold:
            print(f"    drop seg {seg['segment_id']} — max top-1 score {top1_max:.4f} < {score_threshold}")
        elif seg_label in _drop_label_ids:
            print(f"    drop seg {seg['segment_id']} — label is '{HOUSE_ROOM_TYPES[seg_label]}'")
        else:
            final.append(seg)
            continue
        if frame_dir.exists():
            shutil.rmtree(frame_dir)

    with open(output_dir / "segments.json", "w") as f:
        json.dump(final, f, indent=2)
    (output_dir / "boundaries.txt").write_text(
        "\n".join(str(seg["start_frame"]) for seg in final[1:]) + "\n")

    print(
        f"  [ok] {scene_dir.name}: {n} frames → "
        f"{len(segments)} raw → {n_after_trim} after trim → {len(final)} after label+score filter"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label-based segmentation using SigLIP room classification"
    )
    parser.add_argument("--data_root",      required=True,
                        help="Path to reconstructions/ directory")
    parser.add_argument("--output_root",    required=True,
                        help="Root directory for per-scene output")
    parser.add_argument("--batch_size",   type=int, default=32,
                        help="SigLIP inference batch size (default: 32)")
    parser.add_argument("--window_half",  type=int, default=WINDOW_HALF,
                        help=f"Half-size w of voting window 2w+1 (default: {WINDOW_HALF})")
    parser.add_argument("--min_len",         type=int,   default=MIN_SEGMENT_LEN,
                        help=f"Drop segments shorter than this after trimming (default: {MIN_SEGMENT_LEN})")
    parser.add_argument("--score_threshold", type=float, default=SCORE_THRESHOLD,
                        help=f"Drop segments whose max top-1 score never exceeds this (default: {SCORE_THRESHOLD})")
    parser.add_argument("--scene_ids",    nargs="*", default=None,
                        help="Process only these scene IDs (default: all)")
    args = parser.parse_args()

    data_root   = Path(args.data_root)
    output_root = Path(args.output_root)
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    siglip_model = load_siglip_model(device)
    print(f"SigLIP loaded on {device}.\n")

    if args.scene_ids is not None:
        scene_dirs = [data_root / sid for sid in args.scene_ids]
    else:
        scene_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())

    print(f"Processing {len(scene_dirs)} scene(s) ...")
    for scene_dir in scene_dirs:
        print(f"\nScene {scene_dir.name}")
        process_scene(
            scene_dir        = scene_dir,
            output_dir       = output_root / scene_dir.name,
            siglip_model     = siglip_model,
            device           = device,
            batch_size       = args.batch_size,
            window_half      = args.window_half,
            min_len          = args.min_len,
            score_threshold  = args.score_threshold,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
