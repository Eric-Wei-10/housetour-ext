# dataset-stats

Analyzes COLMAP multi-part reconstructions across all scenes in the HouseTour dataset and produces statistical summaries and visualizations.

## What it does

For each scene, the script counts integer-named sub-directories (`0/`, `1/`, `2/`, …) representing COLMAP sub-reconstructions and reads `images.bin` to count registered camera poses per part. It then reports distribution statistics and generates two plots.

## Usage

```bash
python dataset_stats.py [--scenes SCENE_RANGE]
```

`--scenes` accepts ranges, comma-separated IDs, or a mix:

```bash
python dataset_stats.py                        # all scenes
python dataset_stats.py --scenes 1-277
python dataset_stats.py --scenes 7,42,100
python dataset_stats.py --scenes 1-10,20,30-40
```

## Inputs

COLMAP reconstruction hierarchy at:
```
/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions/
  <scene_id>/
    <part_id>/          # integer-named sub-reconstruction
      images.bin
```

## Outputs

All outputs are saved in the same directory as the script:

| File | Description |
|------|-------------|
| `reconstruction_stats.txt` | Per-scene part count and cameras-per-part table, plus summary statistics |
| `reconstruction_hist.png` | Histogram of sub-reconstruction counts across scenes |
| `multipart_cameras.png` | Stacked bar chart of camera poses per part for multi-part scenes |
