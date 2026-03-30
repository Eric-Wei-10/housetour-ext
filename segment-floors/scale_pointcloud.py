"""
Recover the absolute metric scale of a Y-aligned COLMAP reconstruction
using UniDepth V2 metric depth estimation.

The Y-aligned reconstruction (*_aligned.ply + 0_aligned/) produced by
align_pointcloud.py has correct relative geometry but arbitrary scale.
UniDepth V2, given the known camera intrinsics, predicts absolute metric
depth. We estimate the global scale factor s by comparing UniDepth depths
against COLMAP depths at the observed sparse keypoints across frames, then
apply s to the entire reconstruction.

In a single inference pass over all registered frames, this script:
  1. Runs UniDepth V2 on every N-th frame with COLMAP keypoint observations,
     collects depth ratios (UniDepth / COLMAP) to estimate the global scale s
  2. Applies s to *_aligned.ply → *_metric.ply
  3. Applies s to 0_aligned/ → 0_metric/

Scale estimation:
  For each sampled frame, project the COLMAP sparse 3D points (from
  0_aligned/points3D.bin) into the camera → COLMAP depth at each keypoint.
  Compare against UniDepth depth at the same pixel.
  s = median(z_unidepth / z_colmap) over all sampled frames and keypoints.

Inputs:
  reconstructions/<scene>/<scene>_aligned.ply  -- Y-aligned point cloud (arbitrary scale)
  reconstructions/<scene>/0_aligned/           -- Y-aligned COLMAP files
  reconstructions/<scene>/keyframes/           -- RGB images

Outputs:
  reconstructions/<scene>/<scene>_metric.ply   -- metric-scale point cloud
  reconstructions/<scene>/0_metric/            -- metric-scale COLMAP files
    cameras.bin   -- copied unchanged (intrinsics are pixel-based, scale-invariant)
    images.bin    -- camera translations scaled by s (rotations unchanged)
    points3D.bin  -- 3D point positions scaled by s
    scale.txt     -- the estimated scale factor and statistics

Usage:
    python scale_pointcloud.py <scene_id> [--model vits14|vitb14|vitl14] [--skip N]
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
parser = argparse.ArgumentParser()
parser.add_argument("scene_id")
parser.add_argument("--model", default="vits14", choices=["vits14", "vitb14", "vitl14"],
                    help="UniDepth V2 backbone (vits14 is faster, vitl14 is more accurate)")
parser.add_argument("--skip", type=int, default=1,
                    help="Sample every N-th frame for scale estimation (default: 5)")
args = parser.parse_args()

RECON_ROOT     = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SCENE_DIR      = os.path.join(RECON_ROOT, args.scene_id)
COLMAP_ALIGNED = os.path.join(SCENE_DIR, "0_aligned")
COLMAP_METRIC  = os.path.join(SCENE_DIR, "0_metric")
KEYFRAME_DIR   = os.path.join(SCENE_DIR, "keyframes")
PLY_IN         = os.path.join(SCENE_DIR, f"{args.scene_id}_aligned.ply")
PLY_OUT        = os.path.join(SCENE_DIR, f"{args.scene_id}_metric.ply")

os.makedirs(COLMAP_METRIC, exist_ok=True)

# ---------------------------------------------------------------------------
# COLMAP binary readers / writers
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
            iid          = struct.unpack("<I", f.read(4))[0]
            qw,qx,qy,qz = struct.unpack("<4d", f.read(32))
            tx,ty,tz     = struct.unpack("<3d", f.read(24))
            camera_id    = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            pts2d = []
            for _ in range(n2):
                x2, y2 = struct.unpack("<2d", f.read(16))
                pid    = struct.unpack("<q",  f.read(8))[0]
                pts2d.append((x2, y2, pid))
            images[iid] = dict(qw=qw, qx=qx, qy=qy, qz=qz,
                               tx=tx, ty=ty, tz=tz,
                               camera_id=camera_id,
                               name=name.decode(),
                               pts2d=pts2d)
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
            points[pid] = dict(xyz=np.array([x,y,z]), rgb=(r,g,b),
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
# Load COLMAP data
# ---------------------------------------------------------------------------
print("Loading COLMAP data ...")
cameras  = read_cameras_bin(os.path.join(COLMAP_ALIGNED, "cameras.bin"))
images   = read_images_bin (os.path.join(COLMAP_ALIGNED, "images.bin"))
points3d = read_points3d_bin(os.path.join(COLMAP_ALIGNED, "points3D.bin"))

cam       = next(iter(cameras.values()))
W, H      = int(cam["w"]), int(cam["h"])
f, cx, cy = cam["params"][0], cam["params"][1], cam["params"][2]
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
print(f"  Camera: {W}x{H}  f={f:.2f}  cx={cx:.2f}  cy={cy:.2f}")
print(f"  Registered frames : {len(images)}")
print(f"  Sparse 3D points  : {len(points3d)}")

# Pre-build point lookup: point3D_id → xyz in the Y-aligned world frame
pts_xyz = {pid: pt["xyz"] for pid, pt in points3d.items()}

# ---------------------------------------------------------------------------
# Load UniDepth V2
# ---------------------------------------------------------------------------
print(f"\nLoading UniDepth V2 ({args.model}) ...")
try:
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole
except ImportError:
    raise ImportError("Run: pip install git+https://github.com/lpiccinelli-eth/UniDepth.git")

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{args.model}")
model   = model.to(device).eval()
print(f"  Model on {device}")

# K as a numpy array — converted to a fresh tensor per frame to avoid
# in-place corruption by UniDepth's internal crop/resize operations.
K_np = K.astype(np.float32)

# ---------------------------------------------------------------------------
# Single pass: scale estimation (every N-th frame)
# ---------------------------------------------------------------------------
sorted_frames  = sorted(images.values(), key=lambda x: x["name"])
scale_frame_idx = set(range(0, len(sorted_frames), args.skip))  # indices for scale estimation

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

    bgr = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)   # (3,H,W)

    # Create a fresh camera each frame: model.infer mutates the Pinhole's K
    # tensor in-place (crop + resize), so reusing the same object across frames
    # cumulatively corrupts the intrinsics, causing a singular-matrix error.
    camera = Pinhole(K=torch.from_numpy(K_np).to(device))
    with torch.no_grad():
        preds = model.infer(rgb_t, camera)
    depth_m = preds["depth"].squeeze().cpu().numpy()   # (H, W), metric metres

    # --- Collect scale ratios with COLMAP observations ---
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

# ---------------------------------------------------------------------------
# Apply scale to the aligned point cloud → *_metric.ply
# ---------------------------------------------------------------------------
print(f"\nApplying scale to {PLY_IN} ...")
pcd = o3d.io.read_point_cloud(PLY_IN)
pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * scale)
o3d.io.write_point_cloud(PLY_OUT, pcd)
print(f"  Saved → {PLY_OUT}")

# ---------------------------------------------------------------------------
# Apply scale to COLMAP files → 0_metric/
# Rotations are unchanged; only translations and 3D positions scale with s.
#   t_metric = s * t_aligned
#   X_metric = s * X_aligned
# ---------------------------------------------------------------------------
print(f"\nWriting scaled COLMAP files → 0_metric/ ...")

# cameras.bin: intrinsics are pixel-based, unaffected by metric scale
shutil.copy2(os.path.join(COLMAP_ALIGNED, "cameras.bin"),
             os.path.join(COLMAP_METRIC,  "cameras.bin"))
if os.path.exists(os.path.join(COLMAP_ALIGNED, "project.ini")):
    shutil.copy2(os.path.join(COLMAP_ALIGNED, "project.ini"),
                 os.path.join(COLMAP_METRIC,  "project.ini"))

# Save scale factor for reference
with open(os.path.join(COLMAP_METRIC, "scale.txt"), "w") as fh:
    fh.write(f"scale = {scale}\n"
             f"n_correspondences = {len(all_ratios)}\n"
             f"mean  = {np.mean(all_ratios)}\n"
             f"std   = {np.std(all_ratios)}\n")

# images.bin: scale camera translations (rotations are unaffected)
for img in images.values():
    img["tx"] *= scale
    img["ty"] *= scale
    img["tz"] *= scale
write_images_bin(os.path.join(COLMAP_METRIC, "images.bin"), images)
print(f"  images.bin  : {len(images)} poses written")

# points3D.bin: scale 3D point positions
for pt in points3d.values():
    pt["xyz"] = pt["xyz"] * scale
write_points3d_bin(os.path.join(COLMAP_METRIC, "points3D.bin"), points3d)
print(f"  points3D.bin: {len(points3d)} points written")

print(f"\nDone.  Scale factor = {scale:.6f}")
print(f"  Metric PLY     → {PLY_OUT}")
print(f"  Metric COLMAP  → {COLMAP_METRIC}/")
