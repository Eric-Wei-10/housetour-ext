"""
Align a COLMAP scene so that the Y-axis is perpendicular to floors/ceilings.

Reads the original (tilted) point cloud and COLMAP reconstruction.
Automatically detects the dominant floor/ceiling plane via RANSAC, computes
the rotation that maps its normal to [0, 1, 0], and applies it everywhere.

Inputs:
  reconstructions/<scene>/<scene>_pointcloud.ply  -- original tilted point cloud
  reconstructions/<scene>/0/                      -- original COLMAP reconstruction

Outputs:
  reconstructions/<scene>/<scene>_aligned.ply     -- Y-aligned point cloud
  reconstructions/<scene>/0_aligned/              -- Y-aligned COLMAP files
      cameras.bin   -- copied unchanged (intrinsics are camera-relative)
      project.ini   -- copied unchanged
      images.bin    -- camera poses rotated to the Y-aligned frame
      points3D.bin  -- 3D point positions rotated to the Y-aligned frame
      T_align.txt   -- the 4x4 rotation transform for reference

Usage:
    python align_pointcloud.py <scene_id>

Example:
    python align_pointcloud.py 1624
"""

import argparse
import os
import shutil
import struct

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Align a COLMAP scene to the Y-up frame via floor plane detection.")
parser.add_argument("scene_id", help="Scene ID (e.g. 1624)")
args = parser.parse_args()

RECON_ROOT     = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SCENE_DIR      = os.path.join(RECON_ROOT, args.scene_id)
PLY_ORIG       = os.path.join(SCENE_DIR, f"{args.scene_id}_pointcloud.ply")
PLY_ALIGNED    = os.path.join(SCENE_DIR, f"{args.scene_id}_aligned.ply")

def pick_largest_recon(scene_dir):
    """Return the sub-reconstruction subdirectory with the most registered images."""
    best_dir, best_count = None, -1
    for entry in os.scandir(scene_dir):
        if not (entry.is_dir() and entry.name.isdigit()):
            continue
        images_bin = os.path.join(entry.path, "images.bin")
        if not os.path.exists(images_bin):
            continue
        with open(images_bin, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        if n > best_count:
            best_count, best_dir = n, entry.path
    if best_dir is None:
        raise FileNotFoundError(f"No valid COLMAP sub-reconstruction found in {scene_dir}")
    print(f"  Selected sub-reconstruction: {os.path.basename(best_dir)}/ ({best_count} images)")
    return best_dir

COLMAP_DIR     = pick_largest_recon(SCENE_DIR)
COLMAP_ALIGNED = os.path.join(SCENE_DIR, "0_aligned")

os.makedirs(COLMAP_ALIGNED, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and downsample for plane detection
# ---------------------------------------------------------------------------
print(f"Loading point cloud: {PLY_ORIG}")
orig_pcd = o3d.io.read_point_cloud(PLY_ORIG)
print(f"  Points: {len(orig_pcd.points)}")

pcd_down = orig_pcd.voxel_down_sample(voxel_size=0.05)
pts_down = np.asarray(pcd_down.points)

# ---------------------------------------------------------------------------
# Detect the dominant planar surface (floor or ceiling) via RANSAC
# ---------------------------------------------------------------------------
print("\nDetecting dominant plane via RANSAC ...")

# ---------------------------------------------------------------------------
# Compute rotation: plane normal → [0, 1, 0]
# ---------------------------------------------------------------------------
def normal_to_y_rotation(n):
    """Rotation matrix that maps unit vector n to [0, 1, 0]."""
    target = np.array([0., 1., 0.])
    axis   = np.cross(n, target)
    s      = np.linalg.norm(axis)
    if s < 1e-6:
        return np.eye(3) if n[1] > 0 else np.diag([1., -1., 1.])
    axis  /= s
    angle  = np.arccos(np.clip(np.dot(n, target), -1., 1.))
    return Rotation.from_rotvec(angle * axis).as_matrix()

MAX_PLANE_ATTEMPTS = 5
ANGLE_THRESHOLD    = 20.0   # degrees

candidate = pcd_down
normal     = None
inlier_idx = None

for attempt in range(1, MAX_PLANE_ATTEMPTS + 1):
    plane_model, idx = candidate.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=2000)
    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n /= np.linalg.norm(n)
    # Flip so normal points toward +Y
    if np.dot(n, [0., 1., 0.]) < 0:
        n = -n
    angle = np.degrees(np.arccos(np.clip(np.dot(n, [0., 1., 0.]), -1., 1.)))
    print(f"  Attempt {attempt}: normal={np.round(n, 4)}, "
          f"angle from horizontal={angle:.2f}°, inliers={len(idx)}/{len(np.asarray(candidate.points))}")
    if angle <= ANGLE_THRESHOLD:
        normal     = n
        inlier_idx = idx
        break
    # Remove this plane's inliers and try again
    all_idx    = np.arange(len(candidate.points))
    outlier_idx = np.setdiff1d(all_idx, idx)
    candidate  = candidate.select_by_index(outlier_idx)

if normal is None:
    raise ValueError(
        f"Could not find a near-horizontal plane after {MAX_PLANE_ATTEMPTS} attempts "
        f"(threshold: {ANGLE_THRESHOLD}°). Check the point cloud."
    )

print(f"  Accepted plane normal (original frame): {np.round(normal, 4)}")

R_align = normal_to_y_rotation(normal)
euler = Rotation.from_matrix(R_align).as_euler("xyz", degrees=True)
print(f"  Rotation applied (Euler xyz, deg): {np.round(euler, 3)}")

# Compute the floor plane Y coordinate in the aligned frame
# (inlier_idx indexes into `candidate`, which may have had prior planes removed)
inlier_pts     = np.asarray(candidate.select_by_index(inlier_idx).points)
inlier_aligned = (R_align @ inlier_pts.T).T
floor_y        = float(inlier_aligned[:, 1].mean())
print(f"  Floor plane Y (aligned frame): {floor_y:.4f}")

# Save transform for reference (pure rotation, t=0)
T_align = np.eye(4)
T_align[:3, :3] = R_align
np.savetxt(os.path.join(COLMAP_ALIGNED, "T_align.txt"), T_align,
           header="4x4 rigid transform: x_aligned = T @ x_original (homogeneous)")

# Save plane info: normal (in aligned frame = [0,1,0]) and Y offset
np.savetxt(os.path.join(COLMAP_ALIGNED, "floor_plane.txt"),
           np.array([0., 1., 0., -floor_y]),
           header="Floor plane coefficients in aligned frame: a b c d  (ax+by+cz+d=0)")

# ---------------------------------------------------------------------------
# Apply rotation to original point cloud → *_aligned.ply
# ---------------------------------------------------------------------------
pts_orig    = np.asarray(orig_pcd.points)
pts_aligned = (R_align @ pts_orig.T).T
out_pcd = o3d.geometry.PointCloud()
out_pcd.points = o3d.utility.Vector3dVector(pts_aligned)
if orig_pcd.has_colors():
    out_pcd.colors = orig_pcd.colors
o3d.io.write_point_cloud(PLY_ALIGNED, out_pcd)
print(f"\n  Saved aligned point cloud → {PLY_ALIGNED}")

# ---------------------------------------------------------------------------
# Copy cameras.bin and project.ini unchanged
# (intrinsics are camera-relative and unaffected by world-frame rotation)
# ---------------------------------------------------------------------------
for fname in ("cameras.bin", "project.ini"):
    src = os.path.join(COLMAP_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(COLMAP_ALIGNED, fname))
        print(f"  Copied {fname} unchanged.")

# ---------------------------------------------------------------------------
# COLMAP binary helpers
# ---------------------------------------------------------------------------
def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id       = struct.unpack("<I",  f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz     = struct.unpack("<3d", f.read(24))
            camera_id      = struct.unpack("<I",  f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            pts2d = []
            for _ in range(num_pts2d):
                x2, y2     = struct.unpack("<2d", f.read(16))
                point3d_id = struct.unpack("<q",  f.read(8))[0]
                pts2d.append((x2, y2, point3d_id))
            images[image_id] = dict(qw=qw, qx=qx, qy=qy, qz=qz,
                                    tx=tx, ty=ty, tz=tz,
                                    camera_id=camera_id,
                                    name=name.decode(),
                                    pts2d=pts2d)
    return images

def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, img in images.items():
            f.write(struct.pack("<I",  image_id))
            f.write(struct.pack("<4d", img["qw"], img["qx"], img["qy"], img["qz"]))
            f.write(struct.pack("<3d", img["tx"],  img["ty"],  img["tz"]))
            f.write(struct.pack("<I",  img["camera_id"]))
            f.write(img["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", len(img["pts2d"])))
            for x2, y2, pid in img["pts2d"]:
                f.write(struct.pack("<2d", x2, y2))
                f.write(struct.pack("<q",  pid))

def read_points3d_bin(path):
    points = {}
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            pid       = struct.unpack("<Q",  f.read(8))[0]
            x, y, z   = struct.unpack("<3d", f.read(24))
            r, g, b   = struct.unpack("<3B", f.read(3))
            error     = struct.unpack("<d",  f.read(8))[0]
            track_len = struct.unpack("<Q",  f.read(8))[0]
            track = [struct.unpack("<2I", f.read(8)) for _ in range(track_len)]
            points[pid] = dict(xyz=np.array([x, y, z]), rgb=(r, g, b),
                               error=error, track=track)
    return points

def write_points3d_bin(path, points):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, pt in points.items():
            f.write(struct.pack("<Q",  pid))
            f.write(struct.pack("<3d", *pt["xyz"]))
            f.write(struct.pack("<3B", *pt["rgb"]))
            f.write(struct.pack("<d",  pt["error"]))
            f.write(struct.pack("<Q",  len(pt["track"])))
            for img_id, pt2d_idx in pt["track"]:
                f.write(struct.pack("<2I", img_id, pt2d_idx))

# ---------------------------------------------------------------------------
# Transform images.bin: rotate poses into the Y-aligned world frame
#
# Original COLMAP pose: x_cam = R_cam @ x_orig + t_cam
# In Y-aligned frame (t_align = 0):
#   x_aligned = R_align @ x_orig
#   => x_orig  = R_align^T @ x_aligned
# Substituting:
#   x_cam = (R_cam @ R_align^T) @ x_aligned + t_cam
# So:
#   R_new = R_cam @ R_align^T
#   t_new = t_cam   (unchanged, since t_align = 0)
# ---------------------------------------------------------------------------
print("\n  Transforming images.bin ...")
images    = read_images_bin(os.path.join(COLMAP_DIR, "images.bin"))
R_align_T = R_align.T   # R_align is orthogonal: R^{-1} = R^T

for img in images.values():
    R_cam = Rotation.from_quat(
        [img["qx"], img["qy"], img["qz"], img["qw"]]).as_matrix()  # scipy: xyzw order

    R_new = R_cam @ R_align_T
    # t_new = t_cam (translation is unchanged for pure rotation with t_align=0)

    q_new = Rotation.from_matrix(R_new).as_quat()   # [qx, qy, qz, qw]
    img["qx"], img["qy"], img["qz"], img["qw"] = q_new

write_images_bin(os.path.join(COLMAP_ALIGNED, "images.bin"), images)
print(f"  Written {len(images)} poses → 0_aligned/images.bin")

# ---------------------------------------------------------------------------
# Transform points3D.bin: rotate 3D points into the Y-aligned world frame
#   x_aligned = R_align @ x_orig
# ---------------------------------------------------------------------------
print("  Transforming points3D.bin ...")
points3d = read_points3d_bin(os.path.join(COLMAP_DIR, "points3D.bin"))

all_xyz         = np.stack([p["xyz"] for p in points3d.values()])   # (N, 3)
all_xyz_aligned = (R_align @ all_xyz.T).T                            # (N, 3)
for pt, xyz_new in zip(points3d.values(), all_xyz_aligned):
    pt["xyz"] = xyz_new

write_points3d_bin(os.path.join(COLMAP_ALIGNED, "points3D.bin"), points3d)
print(f"  Written {len(points3d)} points → 0_aligned/points3D.bin")

print(f"\nDone. Y-aligned reconstruction → {COLMAP_ALIGNED}")
