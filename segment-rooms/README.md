# segment-rooms

Room segmentation pipeline for house-tour videos. Classifies keyframes by room
type and splits the sequence into per-room segments.

## Pipeline

```
segment_rooms  ── probe_multi_bed ── room_bbox
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

### 2. `probe_multi_bed.py` / `probe_multi_bed.sh`

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

### 3. `room_bbox.py` / `room_bbox.sh`

Offline batch export of oriented room bounding boxes for all segments in a scene.

Per segment:
1. Build XZ density histogram from filtered 3D points (multi-view filter → dense/sparse mode).
2. Threshold histogram to top 20% density bins → wall skeleton.
3. Radon-like projection on skeleton → detect dominant wall orientation.
4. Fix bbox orientation to the detected wall direction, find minimum-area rectangle
   covering 95% of total point weight, constrained to contain the center of mass.

**Outputs** (per scene under `label_segments/<scene_id>/room_bbox/`):

| File | Description |
|---|---|
| `seg_<id>.json` | CoM, dominant wall angle, bbox corners + area |
| `seg_<id>.png` | 2-panel figure: density+bbox / skeleton+direction line |

```bash
sbatch ./segment-rooms/room_bbox.sh 7
sbatch ./segment-rooms/room_bbox.sh 1-50
sbatch ./segment-rooms/room_bbox.sh 7 14 --mode sparse --density_percentile 70
```

### 4. `export_xz_density.py` / `export_xz_density.sh`

Offline batch export of XZ density histograms with uniform-angle bbox search
(36 orientations, no wall detection). Predecessor to `room_bbox.py`.

```bash
sbatch ./segment-rooms/export_xz_density.sh 7
sbatch ./segment-rooms/export_xz_density.sh 1-50
```

## Viewer

### `segment_viewer.py`

Interactive 3D segment viewer built with [viser](https://viser.studio). Renders
one camera frustum per keyframe, coloured by segment ID. Click a frustum to
preview the corresponding keyframe image. Double-click a frustum to show the
segment's filtered point cloud and compute a wall-aligned bounding box (same
algorithm as `room_bbox.py`).

```bash
python segment-rooms/segment_viewer.py --scene_id 7 [--port 8080] [--every_n 2]
python segment-rooms/segment_viewer.py --scene_id 7 --smooth_bbox
python segment-rooms/segment_viewer.py --scene_id 7 --min_obs 2 --min_multi_view 2
```

Online outputs are saved to `label_segments/<scene_id>/viewer_output/`.

If running on a remote cluster:
```bash
ssh -L 8080:localhost:8080 -J <user>@euler.ethz.ch <user>@<compute-node>
```

## Support modules

| File | Role |
|---|---|
| `bbox_utils.py` | Shared bbox logic: `detect_dominant_angle`, `find_bbox_at_angle` |
| `siglip.py` | SigLIP model loading, feature extraction, room classification |
| `utils.py` | COLMAP `images.bin` reader |
| `visualization.py` | Segment grid, similarity matrix, per-frame overlay figures |
