# segment-rooms

Room segmentation pipeline for house-tour videos. Classifies keyframes by room
type and splits the sequence into per-room segments.

## Pipeline

```
                 ┌─ refine_vlm      (approach A)
segment_rooms  ──┤
                 └─ probe_multi_bed (approach B)
```

`refine_vlm` and `probe_multi_bed` are two independent approaches to the same
problem — splitting under-segmented adjacent bedrooms. They are being evaluated
in parallel; only one will be used in the final pipeline.

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

### 3. `probe_multi_bed.py` / `probe_multi_bed.sh`

Second-pass bedroom post-processing. The SigLIP pipeline often fails to separate
two adjacent bedrooms, merging them into one segment. This script uses
Qwen2.5-VL-7B to detect and split such cases.

For each bedroom segment:
1. Ask the VLM per-frame: how many beds are visible, and (if multiple) are they
   in the same room or different rooms?
2. Apply two-pass label smoothing to suppress per-frame noise.
3. Recursively scan for runs of `multiple`-bed frames. For each run, vote using
   only the raw-`multiple` frames: if `different` strictly outvotes `same`, the
   run is a transition boundary — drop those frames and split the segment.
4. Each resulting sub-segment must have ≥ 5 frames to be kept.

`--dry_run` runs the full analysis and prints the log without writing to JSON.

```bash
sbatch ./segment-rooms/probe_multi_bed.sh 7 --dry_run
sbatch ./segment-rooms/probe_multi_bed.sh 7 --out_dir /tmp/probe
sbatch ./segment-rooms/probe_multi_bed.sh 1-200 --out_dir /tmp/probe
```

## Viewer

### `viewer.py`

Interactive 3D segment viewer built with [viser](https://viser.studio). Renders
one camera frustum per keyframe, coloured by segment ID. Click a frustum to
preview the corresponding keyframe image.

> **Note:** The point-cloud highlight on frustum click uses pinhole projection
> with an approximate Z-buffer (depth buffer + dilation) to suppress occluded
> points. Occlusion culling is approximate — sparse areas of the point cloud
> may still let some behind-wall points through.

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
