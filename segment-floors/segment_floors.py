"""
Detect floor levels from camera extrinsic parameters and point cloud geometry.

Instead of relying on the point cloud alone (which may include noisy
outdoor/garden reconstructions), this script combines two signals:
  • camera-center Y values from COLMAP images.bin
  • a voxel-downsampled point-cloud Y histogram

COLMAP convention:  p_cam = R @ p_world + t
Camera center:      C_world = -R^T @ t   (NOT t directly)

The Y-axis is vertical in the 0_metric reconstruction.  Camera centers form
tight horizontal bands at each floor level, unaffected by garden geometry.

Algorithm:
  1. Load metric PLY, voxel-downsample (0.05 m), build fine Y histogram (0.01 m bins)
  2. Smooth + peak-detect → candidate floor-level Y positions
  3. Parse 0_metric/images.bin → camera centers C_world for every keyframe
  4. Build a coarser camera-center Y histogram (--bin_size, default 0.2 m)
  5. Optionally read RANSAC floor plane from floor_plane.txt + scale.txt
  6. Save a two-panel histogram (point-cloud Y / camera Y) with annotations

Inputs (required):
  reconstructions/<scene>/0_metric/images.bin   -- COLMAP binary images file
  reconstructions/<scene>/<scene>_metric.ply    -- metric point cloud

Inputs (optional):
  reconstructions/<scene>/0_metric/floor_plane.txt  -- RANSAC floor plane [a,b,c,d]
  reconstructions/<scene>/0_metric/scale.txt         -- metric scale factor

Output:
  reconstructions/<scene>/camera_y_histogram.png
    two-panel figure: smoothed voxel-Y density (top) and camera-Y counts (bottom),
    with detected peaks and optional RANSAC floor-plane annotation

Usage:
    python segment_floors.py <scene_id> [--bin_size BIN_SIZE]

Example:
    python segment_floors.py 1624
    python segment_floors.py 1624 --bin_size 0.1
"""

import argparse
# import json
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Segment floors from camera extrinsic Y-coordinates.")
parser.add_argument("scene_id", help="Scene ID (e.g. 1624)")
# parser.add_argument(
#     "--y_padding", type=float, default=0.3,
#     help="Extra metres added above/below camera Y range per floor (default 0.3)")
parser.add_argument(
    "--bin_size", type=float, default=0.2,
    help="Histogram bin size in metres (default 0.2)")
args = parser.parse_args()

RECON_ROOT  = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SCENE_DIR   = os.path.join(RECON_ROOT, args.scene_id)
IMAGES_BIN  = os.path.join(SCENE_DIR, "0_metric", "images.bin")
PLY_PATH    = os.path.join(SCENE_DIR, f"{args.scene_id}_metric.ply")

# ---------------------------------------------------------------------------
# Helper: quaternion (w, x, y, z) → 3×3 rotation matrix
# ---------------------------------------------------------------------------
def qvec_to_rotmat(qvec):
    """Convert COLMAP quaternion [w, x, y, z] to a 3×3 rotation matrix."""
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

# ---------------------------------------------------------------------------
# Parse images.bin  (COLMAP binary format)
# Returns list of camera centers in world coordinates
# ---------------------------------------------------------------------------
def read_images_bin(path):
    """
    Parse a COLMAP images.bin file.

    Returns
    -------
    centers : np.ndarray, shape (N, 3)
        Camera centers in world coordinates:  C = -R^T @ t
    names   : list of str
        Image filenames, same order as centers.
    """
    centers = []
    names   = []

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        print(f"  images.bin: {num_images} images")

        for _ in range(num_images):
            # image_id (uint32)
            f.read(4)

            # quaternion qvec: (w, x, y, z)  — 4 × float64
            qvec = np.array(struct.unpack("<4d", f.read(32)))

            # translation tvec  — 3 × float64
            tvec = np.array(struct.unpack("<3d", f.read(24)))

            # camera_id (uint32)
            f.read(4)

            # image name (null-terminated string)
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            names.append(name_bytes.decode("utf-8"))

            # num_points2D  (uint64)
            num_p2d = struct.unpack("<Q", f.read(8))[0]

            # skip point2D entries (each: x float64, y float64, point3D_id int64)
            f.read(num_p2d * 24)

            # Compute camera center in world coordinates
            R = qvec_to_rotmat(qvec)
            C = -R.T @ tvec        # C_world = -R^T t
            centers.append(C)

    return np.array(centers), names


# ---------------------------------------------------------------------------
# Load point cloud Y range
# ---------------------------------------------------------------------------
print(f"\n[1] Loading point cloud from:\n    {PLY_PATH}")
full_pcd = o3d.io.read_point_cloud(PLY_PATH)
print(f"  Total points: {len(full_pcd.points)}")
down_pcd = full_pcd.voxel_down_sample(voxel_size=0.05)
pcd_y    = np.asarray(down_pcd.points)[:, 1]
print(f"  After voxel downsample (0.05 m): {len(down_pcd.points)} points")
y_min, y_max = float(pcd_y.min()), float(pcd_y.max())
print(f"  Point cloud Y range: [{y_min:.3f}, {y_max:.3f}] m  "
      f"(span {y_max - y_min:.3f} m)")

# ---------------------------------------------------------------------------
# Load camera centers
# ---------------------------------------------------------------------------
print(f"\n[2] Parsing camera extrinsics from:\n    {IMAGES_BIN}")
centers, img_names = read_images_bin(IMAGES_BIN)
y_vals = centers[:, 1]
print(f"  Camera Y range: [{y_vals.min():.3f}, {y_vals.max():.3f}] m  "
      f"(span {y_vals.max() - y_vals.min():.3f} m)")

# ---------------------------------------------------------------------------
# Build 1-D histogram of camera Y positions, binned over [y_min, y_max]
# ---------------------------------------------------------------------------
print(f"\n[3] Building camera-Y histogram (bin size = {args.bin_size} m) ...")
n_bins   = int(np.ceil((y_max - y_min) / args.bin_size))
hist_counts, bin_edges = np.histogram(y_vals, bins=n_bins, range=(y_min, y_max))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ---------------------------------------------------------------------------
# Build fine-grained voxel-Y histogram (0.01 m bins), smoothed + peak-detected
# ---------------------------------------------------------------------------
print(f"\n[4] Building voxel-Y histogram (0.01 m bins) ...")
vox_resolution  = 0.01
vox_n_bins      = int(np.ceil((y_max - y_min) / vox_resolution))
vox_counts, vox_edges = np.histogram(pcd_y, bins=vox_n_bins, range=(y_min, y_max))
vox_bin_centers = 0.5 * (vox_edges[:-1] + vox_edges[1:])
vox_smooth      = gaussian_filter1d(vox_counts.astype(float), sigma=2)

vox_min_dist   = int(0.2 / vox_resolution)          # floors at least 0.2 m apart
vox_threshold  = np.percentile(vox_smooth, 90)
vox_peaks, _   = find_peaks(vox_smooth, distance=vox_min_dist, height=vox_threshold)
print(f"  Voxel peaks at Y = {np.round(vox_bin_centers[vox_peaks], 3)} m")

# ---------------------------------------------------------------------------
# Read RANSAC floor plane Y in metric units (aligned_y * scale)
# ---------------------------------------------------------------------------
ransac_floor_y = None
floor_plane_path = os.path.join(SCENE_DIR, "0_metric", "floor_plane.txt")
scale_txt_path   = os.path.join(SCENE_DIR, "0_metric",  "scale.txt")
if os.path.exists(floor_plane_path) and os.path.exists(scale_txt_path):
    coeffs = np.loadtxt(floor_plane_path, comments="#")       # [a, b, c, d]
    floor_y_aligned = float(-coeffs[3])                        # y = -d
    with open(scale_txt_path) as fh:
        scale_factor = float(fh.readline().split("=")[1])      # "scale = <value>"
    ransac_floor_y = floor_y_aligned * scale_factor
    print(f"  RANSAC floor plane Y (metric): {ransac_floor_y:.4f} m  "
          f"(aligned={floor_y_aligned:.4f}, scale={scale_factor:.6f})")
else:
    print("  floor_plane.txt or scale.txt not found — skipping RANSAC annotation.")

# ---------------------------------------------------------------------------
# Save combined histogram: voxel/point cloud (top) + camera (bottom)
# Both share the X axis (Y coordinate = height).
# ---------------------------------------------------------------------------
fig, (ax_vox, ax_cam) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax_vox.plot(vox_bin_centers, vox_smooth, color="darkorange", label="smoothed histogram")
ax_vox.plot(vox_bin_centers[vox_peaks], vox_smooth[vox_peaks],
            "x", color="red", ms=10, label="peaks")
ax_vox.axhline(vox_threshold, color="gray", linestyle="--", linewidth=0.8,
               label="90th-pct threshold")
ax_vox.set_ylabel("Point density")
ax_vox.set_title("Point cloud Y distribution")
ax_vox.legend(fontsize=8)

ax_cam.plot(bin_centers, hist_counts, color="steelblue", label="camera-Y histogram")
ax_cam.set_xlabel("Y (m)")
ax_cam.set_ylabel("Camera count")
ax_cam.set_title("Camera-center Y distribution")
ax_cam.legend(fontsize=8)

if ransac_floor_y is not None:
    textbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
    label = f"RANSAC floor  Y = {ransac_floor_y:.3f} m"
    for ax in (ax_vox, ax_cam):
        ax.axvline(ransac_floor_y, color="red", linestyle="--", linewidth=1.2)
        ax.text(ransac_floor_y, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=8, color="red", bbox=textbox_props)

fig.suptitle(f"Scene {args.scene_id} – floor segmentation", fontsize=13)
fig.tight_layout()

hist_path = os.path.join(SCENE_DIR, "camera_y_histogram.png")
fig.savefig(hist_path, bbox_inches="tight")
plt.close(fig)
print(f"\n  Histogram saved → {hist_path}")
