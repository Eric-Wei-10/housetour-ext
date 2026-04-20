"""
Align a COLMAP scene to Y-up, then recover its absolute metric scale.

Fuses align_pointcloud.py and scale_pointcloud.py into a single pass:
  1. Detect the dominant floor plane via RANSAC, compute R_align so that
     the floor normal maps to [0,1,0].  Apply R_align to the point cloud
     and COLMAP poses/points entirely in memory (no intermediate files).
  2. Run UniDepth V2 on every N-th keyframe to estimate the scale factor s.
     Apply s to the aligned geometry.
  3. Write all outputs once.

Inputs:
  reconstructions/<scene>/<scene>_pointcloud.ply  -- original point cloud
  reconstructions/<scene>/<N>/                    -- original COLMAP reconstruction
                                                     (largest sub-reconstruction selected)
  reconstructions/<scene>/keyframes/              -- RGB images for depth estimation

Outputs:
  reconstructions/<scene>/<scene>_metric.ply      -- metric-scale, Y-aligned point cloud
  reconstructions/<scene>/0_metric/
    cameras.bin     -- copied unchanged (intrinsics are pixel-based)
    project.ini     -- copied unchanged (if present)
    images.bin      -- aligned + scaled camera poses
    points3D.bin    -- aligned + scaled 3D points
    scale.txt       -- estimated scale factor and statistics
    T_align.txt     -- 4×4 alignment rotation for reference
    floor_plane.txt -- floor plane coefficients in aligned frame

Usage:
    python align_scale_pointcloud.py <scene_id> [--model vits14|vitb14|vitl14] [--skip N]
"""

import argparse
import os
import shutil
import struct

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Align to Y-up and recover metric scale in one pass.")
parser.add_argument("scene_id", help="Scene ID (e.g. 1624)")
parser.add_argument("--model", default="vits14", choices=["vits14", "vitb14", "vitl14"],
                    help="UniDepth V2 backbone (vits14 is faster, vitl14 is more accurate)")
parser.add_argument("--skip", type=int, default=1,
                    help="Sample every N-th frame for scale estimation (default: 1)")
args = parser.parse_args()

RECON_ROOT    = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SCENE_DIR     = os.path.join(RECON_ROOT, args.scene_id)
PLY_ORIG      = os.path.join(SCENE_DIR, f"{args.scene_id}_pointcloud.ply")
PLY_OUT       = os.path.join(SCENE_DIR, f"{args.scene_id}_metric.ply")
COLMAP_METRIC = os.path.join(SCENE_DIR, "0_metric")
KEYFRAME_DIR  = os.path.join(SCENE_DIR, "keyframes")

os.makedirs(COLMAP_METRIC, exist_ok=True)

# ---------------------------------------------------------------------------
# COLMAP binary helpers
# ---------------------------------------------------------------------------
def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id   = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w        = struct.unpack("<Q", f.read(8))[0]
            h        = struct.unpack("<Q", f.read(8))[0]
            nparams  = {0:3,1:4,2:4,3:5,4:5,5:8,6:8,7:8,8:12,9:8,10:9}.get(model_id, 4)
            params   = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
            cameras[cam_id] = dict(model_id=model_id, w=w, h=h, params=list(params))
    return cameras

def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid          = struct.unpack("<I",  f.read(4))[0]
            qw,qx,qy,qz = struct.unpack("<4d", f.read(32))
            tx,ty,tz     = struct.unpack("<3d", f.read(24))
            camera_id    = struct.unpack("<I",  f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            n2    = struct.unpack("<Q", f.read(8))[0]
            pts2d = []
            for _ in range(n2):
                x2, y2 = struct.unpack("<2d", f.read(16))
                pid    = struct.unpack("<q",  f.read(8))[0]
                pts2d.append((x2, y2, pid))
            images[iid] = dict(qw=qw, qx=qx, qy=qy, qz=qz,
                               tx=tx, ty=ty, tz=tz,
                               camera_id=camera_id,
                               name=name.decode(), pts2d=pts2d)
    return images

def read_points3d_bin(path):
    points = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid     = struct.unpack("<Q",  f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            error   = struct.unpack("<d",  f.read(8))[0]
            tl      = struct.unpack("<Q",  f.read(8))[0]
            track   = [struct.unpack("<2I", f.read(8)) for _ in range(tl)]
            points[pid] = dict(xyz=np.array([x, y, z]), rgb=(r, g, b),
                               error=error, track=track)
    return points

def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, img in images.items():
            f.write(struct.pack("<I",  iid))
            f.write(struct.pack("<4d", img["qw"], img["qx"], img["qy"], img["qz"]))
            f.write(struct.pack("<3d", img["tx"],  img["ty"],  img["tz"]))
            f.write(struct.pack("<I",  img["camera_id"]))
            f.write(img["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", len(img["pts2d"])))
            for x2, y2, pid in img["pts2d"]:
                f.write(struct.pack("<2d", x2, y2))
                f.write(struct.pack("<q",  pid))

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
# STEP 1 — ALIGN
# ---------------------------------------------------------------------------

def pick_largest_recon(scene_dir):
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

print("=" * 60)
print("STEP 1 — Align to Y-up")
print("=" * 60)

COLMAP_DIR = pick_largest_recon(SCENE_DIR)

print(f"\nLoading point cloud: {PLY_ORIG}")
orig_pcd = o3d.io.read_point_cloud(PLY_ORIG)
print(f"  Points: {len(orig_pcd.points)}")

pcd_down = orig_pcd.voxel_down_sample(voxel_size=0.05)

print("\nDetecting dominant plane via RANSAC ...")

MAX_PLANE_ATTEMPTS = 5
ANGLE_THRESHOLD    = 20.0

candidate  = pcd_down
normal     = None
inlier_idx = None

for attempt in range(1, MAX_PLANE_ATTEMPTS + 1):
    plane_model, idx = candidate.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=2000)
    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n /= np.linalg.norm(n)
    if np.dot(n, [0., 1., 0.]) < 0:
        n = -n
    angle_from_horiz = np.degrees(np.arccos(np.clip(np.dot(n, [0., 1., 0.]), -1., 1.)))
    print(f"  Attempt {attempt}: normal={np.round(n, 4)}, "
          f"angle from horizontal={angle_from_horiz:.2f}°, "
          f"inliers={len(idx)}/{len(np.asarray(candidate.points))}")
    if angle_from_horiz <= ANGLE_THRESHOLD:
        normal     = n
        inlier_idx = idx
        break
    all_idx     = np.arange(len(candidate.points))
    outlier_idx = np.setdiff1d(all_idx, idx)
    candidate   = candidate.select_by_index(outlier_idx)

if normal is None:
    raise ValueError(
        f"Could not find a near-horizontal plane after {MAX_PLANE_ATTEMPTS} attempts "
        f"(threshold: {ANGLE_THRESHOLD}°). Check the point cloud.")

print(f"  Accepted plane normal (original frame): {np.round(normal, 4)}")

R_align = normal_to_y_rotation(normal)
euler   = Rotation.from_matrix(R_align).as_euler("xyz", degrees=True)
print(f"  Rotation applied (Euler xyz, deg): {np.round(euler, 3)}")

inlier_pts     = np.asarray(candidate.select_by_index(inlier_idx).points)
inlier_aligned = (R_align @ inlier_pts.T).T
floor_y        = float(inlier_aligned[:, 1].mean())
print(f"  Floor plane Y (aligned frame): {floor_y:.4f}")

# Apply R_align to point cloud (in memory)
pts_aligned = (R_align @ np.asarray(orig_pcd.points).T).T
pcd_aligned = o3d.geometry.PointCloud()
pcd_aligned.points = o3d.utility.Vector3dVector(pts_aligned)
if orig_pcd.has_colors():
    pcd_aligned.colors = orig_pcd.colors

# Align COLMAP poses in memory: R_new = R_cam @ R_align^T, t unchanged
print("\n  Transforming images.bin ...")
images    = read_images_bin(os.path.join(COLMAP_DIR, "images.bin"))
R_align_T = R_align.T

for img in images.values():
    R_cam = Rotation.from_quat(
        [img["qx"], img["qy"], img["qz"], img["qw"]]).as_matrix()
    R_new = R_cam @ R_align_T
    q_new = Rotation.from_matrix(R_new).as_quat()   # [qx, qy, qz, qw]
    img["qx"], img["qy"], img["qz"], img["qw"] = q_new

print(f"  {len(images)} poses aligned.")

# Align sparse 3D points in memory: X_aligned = R_align @ X_orig
print("  Transforming points3D.bin ...")
points3d = read_points3d_bin(os.path.join(COLMAP_DIR, "points3D.bin"))

all_xyz         = np.stack([p["xyz"] for p in points3d.values()])
all_xyz_aligned = (R_align @ all_xyz.T).T
for pt, xyz_new in zip(points3d.values(), all_xyz_aligned):
    pt["xyz"] = xyz_new

print(f"  {len(points3d)} sparse points aligned.")

# ---------------------------------------------------------------------------
# STEP 2 — SCALE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — Recover metric scale via UniDepth V2")
print("=" * 60)

# Read cameras directly from the original COLMAP dir (unchanged by alignment)
cameras       = read_cameras_bin(os.path.join(COLMAP_DIR, "cameras.bin"))
cam           = next(iter(cameras.values()))
W, H          = int(cam["w"]), int(cam["h"])
f_cam, cx, cy = cam["params"][0], cam["params"][1], cam["params"][2]
K    = np.array([[f_cam, 0, cx], [0, f_cam, cy], [0, 0, 1]], dtype=np.float64)
K_np = K.astype(np.float32)
print(f"\n  Camera: {W}x{H}  f={f_cam:.2f}  cx={cx:.2f}  cy={cy:.2f}")
print(f"  Registered frames : {len(images)}")
print(f"  Sparse 3D points  : {len(points3d)}")

# Point lookup (aligned positions, before scaling)
pts_xyz = {pid: pt["xyz"] for pid, pt in points3d.items()}

print(f"\nLoading UniDepth V2 ({args.model}) ...")
try:
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole
except ImportError:
    raise ImportError("Run: pip install git+https://github.com/lpiccinelli-eth/UniDepth.git")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{args.model}")
model  = model.to(device).eval()
print(f"  Model on {device}")

sorted_frames   = sorted(images.values(), key=lambda x: x["name"])
scale_frame_idx = set(range(0, len(sorted_frames), args.skip))

print(f"\nRunning UniDepth on {len(scale_frame_idx)} frame(s) "
      f"(every {args.skip}-th of {len(sorted_frames)}) ...")

all_ratios = []

for idx, img_meta in enumerate(tqdm(sorted_frames)):
    rgb_path = os.path.join(KEYFRAME_DIR, img_meta["name"])
    if not os.path.exists(rgb_path):
        tqdm.write(f"  [skip] RGB not found: {img_meta['name']}")
        continue

    if idx not in scale_frame_idx:
        continue

    bgr   = cv2.imread(rgb_path)
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)

    # Create a fresh camera each frame: model.infer mutates the Pinhole's K
    # tensor in-place (crop + resize), so reusing the same object across frames
    # cumulatively corrupts the intrinsics, causing a singular-matrix error.
    camera = Pinhole(K=torch.from_numpy(K_np).to(device))
    with torch.no_grad():
        preds = model.infer(rgb_t, camera)
    depth_m = preds["depth"].squeeze().cpu().numpy()

    valid_obs = [(u, v, pid) for u, v, pid in img_meta["pts2d"] if pid >= 0]
    if len(valid_obs) < 10:
        continue

    R_wc = Rotation.from_quat([img_meta["qx"], img_meta["qy"],
                                img_meta["qz"], img_meta["qw"]]).as_matrix()
    t_wc = np.array([img_meta["tx"], img_meta["ty"], img_meta["tz"]])

    for u, v, pid in valid_obs:
        if pid not in pts_xyz:
            continue
        X_cam    = R_wc @ pts_xyz[pid] + t_wc
        z_colmap = X_cam[2]
        if z_colmap <= 0:
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        z_uni = float(depth_m[vi, ui])
        if z_uni <= 0 or z_uni > 20.0:
            continue
        all_ratios.append(z_uni / z_colmap)

if not all_ratios:
    raise RuntimeError("No valid depth correspondences found. "
                       "Check keyframe paths and COLMAP data.")

scale = float(np.median(all_ratios))
print(f"\n  Depth ratios collected : {len(all_ratios)}")
print(f"  Scale factor  s = {scale:.6f}  "
      f"(mean={np.mean(all_ratios):.4f}, std={np.std(all_ratios):.4f})")

# Apply scale to geometry (in memory)
pcd_aligned.points = o3d.utility.Vector3dVector(
    np.asarray(pcd_aligned.points) * scale)

for img in images.values():
    img["tx"] *= scale
    img["ty"] *= scale
    img["tz"] *= scale

for pt in points3d.values():
    pt["xyz"] = pt["xyz"] * scale

# ---------------------------------------------------------------------------
# Write all outputs once
# ---------------------------------------------------------------------------
print(f"\nWriting outputs → {COLMAP_METRIC}/ ...")

o3d.io.write_point_cloud(PLY_OUT, pcd_aligned)
print(f"  {PLY_OUT}")

for fname in ("cameras.bin", "project.ini"):
    src = os.path.join(COLMAP_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(COLMAP_METRIC, fname))

T_align = np.eye(4)
T_align[:3, :3] = R_align
np.savetxt(os.path.join(COLMAP_METRIC, "T_align.txt"), T_align,
           header="4x4 rigid transform: x_aligned = T @ x_original (homogeneous)")
np.savetxt(os.path.join(COLMAP_METRIC, "floor_plane.txt"),
           np.array([0., 1., 0., -floor_y]),
           header="Floor plane coefficients in aligned frame: a b c d  (ax+by+cz+d=0)")

with open(os.path.join(COLMAP_METRIC, "scale.txt"), "w") as fh:
    fh.write(f"scale = {scale}\n"
             f"n_correspondences = {len(all_ratios)}\n"
             f"mean  = {np.mean(all_ratios)}\n"
             f"std   = {np.std(all_ratios)}\n")

write_images_bin(os.path.join(COLMAP_METRIC, "images.bin"), images)
write_points3d_bin(os.path.join(COLMAP_METRIC, "points3D.bin"), points3d)

print(f"  {COLMAP_METRIC}/cameras.bin  project.ini  T_align.txt  floor_plane.txt")
print(f"  {COLMAP_METRIC}/scale.txt  images.bin  points3D.bin")
print(f"\nDone.  Scale factor = {scale:.6f}")
