# segment-rooms

Room segmentation pipeline for house-tour videos. Classifies keyframes by room
type and splits the sequence into per-room segments.

## Pipeline

```
segment_rooms  →  refine_vlm
```

### 1. `segment_rooms.py` / `segment_rooms.sh`

SigLIP-based room segmentation. Embeds each keyframe with SigLIP, classifies it
into one of the 14 room types via text–image similarity, applies a sliding-window
majority vote to smooth labels, and splits the sequence at label-change boundaries.

**Outputs** (per scene under `label_segments/<scene_id>/`):

| File | Description |
|---|---|
| `siglip_features.npy` | float32 `(N, D)` L2-normalised SigLIP embeddings |
| `room_labels.npy` | int32 `(N,)` raw per-frame room label |
| `room_labels_smoothed.npy` | int32 `(N,)` smoothed per-frame room label |
| `room_scores.npy` | float32 `(N, C)` per-frame per-category scores |
| `segments.json` | list of segment dicts |
| `boundaries.txt` | detected boundary frame indices |
| `segment_grid.png` | thumbnail grid of representative frames per segment |

```bash
sbatch ./segment-rooms/segment_rooms.sh 14
sbatch ./segment-rooms/segment_rooms.sh 11 14 1002
sbatch ./segment-rooms/segment_rooms.sh 100-200
sbatch ./segment-rooms/segment_rooms.sh 11 100-200 --batch_size 64
```

### 2. `refine_vlm.py` / `refine_vlm.sh`

VLM-based post-processing that detects and splits under-segmented bedroom segments.
Uses Qwen2.5-VL-7B with the prior that each bedroom contains exactly one unique bed.

For each bedroom segment:
1. Filter all frames to those where a bed is clearly visible (one frame at a time).
2. Evenly sample up to `n_frames` bed-visible frames, preserving temporal order.
3. Ask the VLM whether all sampled frames show the **same** bed.
4. If different, ask where the bed first changes and split there (midpoint between
   the last old-bed sample and the first new-bed sample).
5. Recurse up to `max_depth` levels.

Modifies `segments.json` in-place. Sub-segment IDs follow the pattern
`5 → 5_0, 5_1 → 5_0_0, 5_0_1`, etc.

```bash
sbatch ./segment-rooms/refine_vlm.sh 14
sbatch ./segment-rooms/refine_vlm.sh 11 14 1002
sbatch ./segment-rooms/refine_vlm.sh 100-200
sbatch ./segment-rooms/refine_vlm.sh 14 --n_frames 12 --dry_run
```

## Viewer

### `viewer.py`

Interactive 3D segment viewer built with [viser](https://viser.studio). Renders
one camera frustum per keyframe, coloured by segment ID. Click a frustum to
preview the corresponding keyframe image.

> **Note:** The point-cloud highlight on frustum click uses a simple pinhole
> visibility check (projects all points and keeps those inside the image plane).
> It does **not** filter occluded points — a point behind geometry may still be
> highlighted if it projects inside the frustum bounds.

```bash
python segment-rooms/viewer.py --scene_id 7 [--port 8080] [--every_n 2]
```

If running on a remote cluster:
```bash
ssh -L 8080:localhost:8080 -J <user>@euler.ethz.ch <user>@<compute-node>
```

## Support modules

| File | Role |
|---|---|
| `siglip.py` | SigLIP model loading, feature extraction, room classification |
| `utils.py` | COLMAP `images.bin` reader |
| `visualization.py` | Segment grid, similarity matrix, per-frame overlay figures |
