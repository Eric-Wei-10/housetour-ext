# segment-floors

Segments multi-floor house reconstructions by detecting individual floor levels from camera positions and point cloud geometry. Processing is two-stage: (1) align the point cloud to Y-up and recover metric scale, then (2) cluster cameras into floor levels.

## Pipeline

```
reconstructions/<scene>/
  <scene>_pointcloud.ply   ──┐
  <N>/images.bin             ├─► align_scale_pointcloud.py ──► <scene>_metric.ply
  keyframes/*.jpg            ┘                                   0_metric/{images,points3D,cameras}.bin
                                                                 0_metric/{T_align,floor_plane,scale}.txt

  <scene>_metric.ply        ─── segment_floors.py ──► floors/histogram.png
  0_metric/images.bin
```

**align_scale_pointcloud.py** — aligns the point cloud to Y-up via dominant-plane RANSAC, then estimates absolute metric scale by comparing UniDepth V2 monocular depth against COLMAP triangulated depth across keyframes.

**segment_floors.py** — builds a histogram of camera-center Y-coordinates and point-cloud Y-values, smooths with a Gaussian filter, and detects floor-level peaks (minimum separation 0.2 m).

## Usage

### SLURM batch (Euler cluster)

```bash
sbatch segment_floors.sh <scenes> [options]

# examples
sbatch segment_floors.sh 1624
sbatch segment_floors.sh 11 14 1002
sbatch segment_floors.sh 100-200
sbatch segment_floors.sh 11 100-200 1624 --model vitl14 --skip 3 --bin_size 0.1
```

SLURM config: 4 CPUs, 64 GB RAM, 1 GPU, 8 h walltime. Logs → `logs/segment_floors_<JOBID>.{out,err}`.

### Individual scripts

```bash
python align_scale_pointcloud.py <scene_id> [--model {vits14|vitb14|vitl14}] [--skip N]
python segment_floors.py         <scene_id> [--bin_size FLOAT]
```

## Options

| Flag | Script | Default | Description |
|------|--------|---------|-------------|
| `--model` | align | `vits14` | UniDepth V2 backbone (`vits14` fastest, `vitl14` most accurate) |
| `--skip N` | align | `1` | Use every N-th keyframe for scale estimation |
| `--bin_size` | segment | `0.2` | Histogram bin size in metres for camera-Y clustering |

## Inputs

| Path | Description |
|------|-------------|
| `reconstructions/<scene>/<scene>_pointcloud.ply` | Original point cloud |
| `reconstructions/<scene>/<N>/images.bin` | COLMAP binary poses (largest sub-reconstruction) |
| `reconstructions/<scene>/keyframes/` | RGB images for depth estimation |

## Outputs

| Path | Description |
|------|-------------|
| `<scene>_metric.ply` | Metric-scale, Y-aligned point cloud |
| `0_metric/images.bin` | Aligned + scaled camera poses |
| `0_metric/points3D.bin` | Aligned + scaled sparse points |
| `0_metric/T_align.txt` | 4×4 rigid transform (`x_aligned = T @ x_original`) |
| `0_metric/floor_plane.txt` | Floor plane coefficients `[a b c d]` in aligned frame |
| `0_metric/scale.txt` | Estimated scale factor and statistics |
| `floors/histogram.png` | Point-cloud Y and camera-Y distributions with detected floor peaks |
