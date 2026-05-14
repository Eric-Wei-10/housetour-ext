"""
Interactive segment viewer (viser-based, works on remote/headless servers).

Renders camera frustums in 3D colored by segment ID.
Click any frustum to preview the corresponding keyframe in the side panel.

Usage:
    python segment-rooms/viewer.py --scene_id 7 [--port 8080]

Then open http://localhost:8080 in your browser.
If running on a remote cluster (e.g. Euler), forward the port via jump host:
    ssh -L 8080:<login-node>:8080 <user>@euler.ethz.ch on login node
    or
    ssh -L 8080:localhost:8080 -J <user>@euler.ethz.ch <user>@<compute-node> on compute node
"""

import argparse
import colorsys
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import viser

HOUSE_ROOM_TYPES = [
    "living room", "bedroom", "bathroom", "kitchen", "dining room",
    "hallway", "office", "study", "garage", "laundry room",
    "storage room", "balcony", "garden", "empty room",
]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--scene_id",  required=True)
parser.add_argument("--port",      type=int, default=8080)
parser.add_argument("--data_root", default="/cluster/project/cvg/students/xinwei/"
                                           "official-housetour-dataset/reconstructions")
parser.add_argument("--seg_root",  default="/cluster/project/cvg/students/xinwei/"
                                           "official-housetour-dataset/label_segments")
parser.add_argument("--every_n",   type=int, default=1,
                    help="Show every N-th frustum (use >1 to reduce clutter)")
args = parser.parse_args()

SCENE_DIR = Path(args.data_root) / args.scene_id
SEG_DIR   = Path(args.seg_root)  / args.scene_id
KF_DIR    = SCENE_DIR / "keyframes"
SEG_JSON   = SEG_DIR / "segments.json"
LABELS_NPY = SEG_DIR / "room_labels.npy"

for candidate in ("0_aligned", "0_metric", "0"):
    IMAGES_BIN = SCENE_DIR / candidate / "images.bin"
    if IMAGES_BIN.exists():
        print(f"Using poses from {candidate}/images.bin")
        break
else:
    raise FileNotFoundError(f"No images.bin found under {SCENE_DIR}")

CAMERAS_BIN = IMAGES_BIN.parent / "cameras.bin"

# ---------------------------------------------------------------------------
# COLMAP reader
# ---------------------------------------------------------------------------
def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid            = struct.unpack("<I",  f.read(4))[0]
            qw,qx,qy,qz   = struct.unpack("<4d", f.read(32))
            tx,ty,tz       = struct.unpack("<3d", f.read(24))
            cam_id         = struct.unpack("<I",  f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n2)
            images[iid] = dict(name=name.decode(),
                               qvec=np.array([qw, qx, qy, qz]),
                               tvec=np.array([tx, ty, tz]),
                               cam_id=cam_id)
    return sorted(images.values(), key=lambda x: x["name"])

# ---------------------------------------------------------------------------
# COLMAP cameras reader
# ---------------------------------------------------------------------------
# Maps COLMAP model_id → number of parameter doubles in cameras.bin
_CAM_MODEL_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}

def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid      = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width    = struct.unpack("<Q", f.read(8))[0]
            height   = struct.unpack("<Q", f.read(8))[0]
            np_      = _CAM_MODEL_NPARAMS.get(model_id, 4)
            params   = struct.unpack(f"<{np_}d", f.read(8 * np_))
            # model 1 (PINHOLE): fx fy cx cy
            # all others: f cx cy [distortion…]  — use f for both fx and fy
            if model_id == 1:
                fx, fy, cx, cy = params[:4]
            else:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            cameras[cid] = dict(width=int(width), height=int(height),
                                fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def camera_center(qvec, tvec):
    qw, qx, qy, qz = qvec
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return -R.T @ tvec

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def load_image_array(img_path, max_size=(640, 480)):
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(max_size, Image.LANCZOS)
        return np.array(img)
    except Exception:
        return np.zeros((480, 640, 3), dtype=np.uint8)

def make_segment_palette(segment_ids):
    """Assign a distinct HSV-based color to each segment ID."""
    sorted_ids = sorted(segment_ids)
    n = len(sorted_ids)
    palette = {}
    for i, sid in enumerate(sorted_ids):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.90)
        palette[sid] = (int(r * 255), int(g * 255), int(b * 255))
    return palette

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def room_name(label_idx):
    if 0 <= label_idx < len(HOUSE_ROOM_TYPES):
        return HOUSE_ROOM_TYPES[label_idx]
    return "unknown"

def visible_point_mask(pts: np.ndarray, qvec, tvec, width, height, fx, fy, cx, cy) -> np.ndarray:
    """Boolean mask of pts (N,3) that project inside this camera's image plane.

    Uses pinhole projection (P_cam = R @ P_world + t) and ignores lens distortion,
    which is fine for an approximate visibility highlight.
    """
    qw, qx, qy, qz = qvec
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    pts_cam = pts @ R.T + tvec              # (N, 3) — world → camera space
    z       = pts_cam[:, 2]
    in_front = z > 0.05
    safe_z   = np.where(in_front, z, 1.0)  # avoid division by near-zero
    u = fx * pts_cam[:, 0] / safe_z + cx
    v = fy * pts_cam[:, 1] / safe_z + cy
    return in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading poses ...")
entries = read_images_bin(IMAGES_BIN)
print(f"  {len(entries)} registered frames")

cameras: dict = {}
if CAMERAS_BIN.exists():
    cameras = read_cameras_bin(CAMERAS_BIN)
    print(f"  {len(cameras)} camera model(s) loaded from cameras.bin")

print("Loading room labels ...")
room_labels = np.load(LABELS_NPY)   # shape (N,), one label per keyframe in sequence order
name_to_label: dict[str, int] = {}
label_idx = 0
for entry in entries:
    if (KF_DIR / entry["name"]).exists():
        if label_idx < len(room_labels):
            name_to_label[entry["name"]] = int(room_labels[label_idx])
        label_idx += 1

print("Loading segments ...")
with open(SEG_JSON) as f:
    segments = json.load(f)
print(f"  {len(segments)} segments")

name_to_seg = {
    fname: seg["segment_id"]
    for seg in segments
    for fname in seg["frame_names"]
}

seg_ids = [int(seg["segment_id"]) for seg in segments]
SEG_PALETTE = make_segment_palette(seg_ids)

# Majority room label per segment (for legend)
seg_room_label: dict[int, int] = {}
for seg in segments:
    frame_labels = [name_to_label[fn] for fn in seg["frame_names"] if fn in name_to_label]
    sid = int(seg["segment_id"])
    if frame_labels:
        counts = np.bincount(frame_labels, minlength=len(HOUSE_ROOM_TYPES))
        seg_room_label[sid] = int(counts.argmax())
    else:
        seg_room_label[sid] = -1

# ---------------------------------------------------------------------------
# Start viser
# ---------------------------------------------------------------------------
server = viser.ViserServer(port=args.port)
print(f"\nViser server running at http://localhost:{args.port}")
print("(If on a remote cluster: ssh -L "
      f"{args.port}:localhost:{args.port} -J <user>@euler.ethz.ch <user>@<compute-node>)\n")

# ---------------------------------------------------------------------------
# GUI side panel
# ---------------------------------------------------------------------------
with server.gui.add_folder("Scene info"):
    server.gui.add_markdown(
        f"**Scene:** {args.scene_id}  \n"
        f"**Frames:** {len(entries)}  \n"
        f"**Segments:** {len(segments)}"
    )

with server.gui.add_folder("Selected frame"):
    gui_frame_label = server.gui.add_html("<i>Click a frustum in the 3D view</i>")
    gui_image       = server.gui.add_image(
        np.zeros((270, 480, 3), dtype=np.uint8), label=None, format="jpeg")

# Segment legend
with server.gui.add_folder("Segments"):
    seg_legend_html = "".join(
        f'<div><span style="color:{rgb_to_hex(SEG_PALETTE[seg["segment_id"]])}">&#9632;</span> '
        f'<b>seg {seg["segment_id"]}</b> &nbsp;'
        f'<i>{room_name(seg_room_label[seg["segment_id"]])}</i>'
        f' — {seg["n_frames"]} frames</div>'
        for seg in segments
    )
    server.gui.add_html(seg_legend_html)

# ---------------------------------------------------------------------------
# Add camera frustums
# ---------------------------------------------------------------------------
print("Adding camera frustums ...")

for i, entry in enumerate(entries):
    sid = name_to_seg.get(entry["name"], -1)
    if sid == -1:
        continue  # dropped frame — not part of any segment

    color  = SEG_PALETTE[sid]
    center = camera_center(entry["qvec"], entry["tvec"])
    qw, qx, qy, qz = entry["qvec"]

    if i % args.every_n != 0:
        continue

    # COLMAP qvec is R_cw (world→camera); viser wxyz expects R_wc (camera→world).
    # Invert by conjugating the quaternion: (qw, -qx, -qy, -qz).
    handle = server.scene.add_camera_frustum(
        name     = f"/cameras/{entry['name']}",
        fov      = float(np.deg2rad(60)),
        aspect   = 4 / 3,
        scale    = 0.08,
        color    = color,
        wxyz     = (qw, -qx, -qy, -qz),
        position = tuple(center),
    )

    def make_click_handler(name, path, segment_id, label_idx, qvec, tvec, cam_id):
        def on_click(_):
            color_hex = rgb_to_hex(SEG_PALETTE[segment_id])
            gui_frame_label.content = (
                f'<span style="color:{color_hex}">&#9632;</span> '
                f'<b>{name}</b><br>'
                f'Segment: <b>seg {segment_id}</b><br>'
                f'Room: <b>{room_name(label_idx)}</b>'
            )
            gui_image.image = load_image_array(path)
            # Highlight points visible from this camera
            if pcd_loaded and cam_id in cameras:
                cam = cameras[cam_id]
                mask = visible_point_mask(
                    pts_all, qvec, tvec,
                    cam["width"], cam["height"],
                    cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                )
                server.scene.add_point_cloud(
                    name="point_cloud/hi",
                    points=pts_all[mask],
                    colors=clrs_all[mask],
                    point_size=0.014)
        return on_click

    handle.on_click(make_click_handler(
        entry["name"], KF_DIR / entry["name"], sid,
        name_to_label.get(entry["name"], -1),
        entry["qvec"], entry["tvec"], entry["cam_id"]))

print(f"Added {len(entries) // args.every_n} frustums.")

# ---------------------------------------------------------------------------
# Optional: load point cloud
# ---------------------------------------------------------------------------
# Kept at module level so frustum click handlers can reference them without recapturing.
pts_all:  np.ndarray | None = None
clrs_all: np.ndarray | None = None
pcd_loaded = False

for ply_name in (f"{args.scene_id}_metric.ply", f"{args.scene_id}_pointcloud.ply"):
    ply_path = SCENE_DIR / ply_name
    if ply_path.exists():
        try:
            import open3d as o3d
            pcd  = o3d.io.read_point_cloud(str(ply_path))
            pts  = np.asarray(pcd.points, dtype=np.float64)   # (N, 3) world coords
            clrs = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() \
                   else np.full((len(pts), 3), 180, dtype=np.uint8)  # (N, 3) RGB 0-255
            if len(pts) > 900_000:
                idx  = np.random.choice(len(pts), 900_000, replace=False)
                pts, clrs = pts[idx], clrs[idx]
            pts_all   = pts
            clrs_all  = clrs
            pcd_loaded = True
            # Background layer: dimmed to 30% so highlighted points stand out
            clrs_dim = (clrs * 0.30).astype(np.uint8)
            server.scene.add_point_cloud(
                name="point_cloud/bg", points=pts, colors=clrs_dim, point_size=0.008)
            # Highlight layer: starts empty, updated on each frustum click
            server.scene.add_point_cloud(
                name="point_cloud/hi",
                points=np.zeros((0, 3), dtype=np.float64),
                colors=np.zeros((0, 3), dtype=np.uint8),
                point_size=0.014)
            print(f"Loaded point cloud: {ply_name} ({len(pts)} pts)")
        except Exception as e:
            print(f"Could not load point cloud: {e}")
        break

# ---------------------------------------------------------------------------
# Keep alive
# ---------------------------------------------------------------------------
print("\nViewer ready. Press Ctrl+C to stop.\n")
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down.")
