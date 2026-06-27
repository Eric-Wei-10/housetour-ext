"""
Interactive floor visualiser – viser  (works on headless / remote cluster)

Segments the metric point cloud into floors via camera-Y histogram peaks,
then shows each floor as a separate coloured point cloud with a GUI-controlled
vertical "explosion" offset so all floors are visible simultaneously.

Usage:
    python viser_floors.py <scene_id> [--port 8080]

SSH tunnel from local machine (on compute node):
    ssh -L 8080:localhost:8080 -J <user>@euler.ethz.ch <user>@<compute-node>
"""

import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import viser
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECON_ROOT = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"

_FLOOR_PALETTE = np.array([
    [ 70, 130, 180],   # steel blue
    [ 60, 179, 113],   # medium sea green
    [255, 160,  20],   # orange
    [186,  85, 211],   # orchid
    [220,  20,  60],   # crimson
], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("scene_id")
parser.add_argument("--port",          type=int,   default=8080)
parser.add_argument("--bin_size",      type=float, default=0.05,
                    help="Camera-Y histogram bin size in m (default 0.05)")
parser.add_argument("--min_floor_sep", type=float, default=1.5,
                    help="Minimum floor-to-floor height in m (default 1.5)")
parser.add_argument("--init_sep",      type=float, default=5.0,
                    help="Initial vertical separation between floors (default 5 m)")
args = parser.parse_args()

SCENE_DIR   = Path(RECON_ROOT) / args.scene_id
PLY_PATH    = SCENE_DIR / f"{args.scene_id}_metric.ply"
IMAGES_BIN  = SCENE_DIR / "0_metric" / "images.bin"
CAMERAS_BIN = SCENE_DIR / "0_metric" / "cameras.bin"
KF_DIR      = SCENE_DIR / "keyframes"

# ---------------------------------------------------------------------------
# COLMAP helpers
# ---------------------------------------------------------------------------
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
            fx = fy = params[0]
            if model_id == 1:
                fx, fy = params[0], params[1]
            cameras[cid] = dict(
                width=int(width), height=int(height),
                fov=2 * np.arctan(width / (2 * fx)),
                aspect=width / height,
            )
    return cameras

def read_images_bin(path):
    entries = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz     = struct.unpack("<3d", f.read(24))
            cam_id         = struct.unpack("<I",  f.read(4))[0]
            name_b = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name_b += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n2)
            qvec = np.array([qw, qx, qy, qz])
            tvec = np.array([tx, ty, tz])
            R    = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            entries.append(dict(
                name=name_b.decode(),
                qvec=qvec, tvec=tvec,
                cam_id=cam_id,
                center=(-R.T @ tvec).astype(np.float32),
            ))
    return sorted(entries, key=lambda x: x["name"])

def load_ransac_floor_y():
    fp = SCENE_DIR / "0_metric" / "floor_plane.txt"
    sp = SCENE_DIR / "0_metric" / "scale.txt"
    if not (fp.exists() and sp.exists()):
        return None
    coeffs = np.loadtxt(fp, comments="#")
    with open(sp) as fh:
        scale = float(fh.readline().split("=")[1])
    return float(-coeffs[3]) * scale

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def height_colors_u8(y_vals, cmap_name="plasma"):
    import matplotlib
    lo, hi = np.percentile(y_vals, 2), np.percentile(y_vals, 98)
    if hi == lo: hi = lo + 1e-3
    norm = np.clip((y_vals - lo) / (hi - lo), 0, 1)
    return (matplotlib.colormaps[cmap_name](norm)[:, :3] * 255).astype(np.uint8)

def load_image_array(path, max_size=(640, 480)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(max_size, Image.LANCZOS)
        return np.array(img)
    except Exception:
        return np.zeros((480, 640, 3), dtype=np.uint8)

# ---------------------------------------------------------------------------
# 1. Load point cloud
# ---------------------------------------------------------------------------
print(f"\n[1] Loading point cloud: {PLY_PATH}")
pcd_full = o3d.io.read_point_cloud(str(PLY_PATH))
pts_all  = np.asarray(pcd_full.points, dtype=np.float32)
rgb_all  = (np.asarray(pcd_full.colors) * 255).astype(np.uint8) \
           if pcd_full.has_colors() \
           else np.full((len(pts_all), 3), 180, dtype=np.uint8)
print(f"    {len(pts_all):,} points")

# ---------------------------------------------------------------------------
# 2. Load camera poses
# ---------------------------------------------------------------------------
print(f"\n[2] Loading camera poses: {IMAGES_BIN}")
entries     = read_images_bin(IMAGES_BIN)
cameras_bin = read_cameras_bin(CAMERAS_BIN) if CAMERAS_BIN.exists() else {}
cam_centers = np.array([e["center"] for e in entries], dtype=np.float32)
cam_y       = cam_centers[:, 1]
print(f"    {len(entries)} cameras  Y∈[{cam_y.min():.3f}, {cam_y.max():.3f}] m")

# ---------------------------------------------------------------------------
# 3. Detect floor levels from camera-Y histogram
# ---------------------------------------------------------------------------
pcd_y      = pts_all[:, 1]
y_min_glob = float(min(pcd_y.min(), cam_y.min()))
y_max_glob = float(max(pcd_y.max(), cam_y.max()))

n_bins = int(np.ceil((y_max_glob - y_min_glob) / args.bin_size))
counts, edges = np.histogram(cam_y, bins=n_bins, range=(y_min_glob, y_max_glob))
bin_c  = 0.5 * (edges[:-1] + edges[1:])
sigma  = max(1, int(0.2 / args.bin_size))
smooth = gaussian_filter1d(counts.astype(float), sigma=sigma)

min_dist = max(1, int(args.min_floor_sep / args.bin_size))
peaks, _ = find_peaks(smooth, distance=min_dist, height=smooth.max() * 0.05)
floor_peak_y = np.sort(bin_c[peaks])
n_floors     = len(floor_peak_y)
print(f"\n[3] {n_floors} floor(s) detected at Y = {np.round(floor_peak_y, 3)} m")

ransac_y = load_ransac_floor_y()

if ransac_y is not None:
    # Nearest peak above the RANSAC plane → ground-floor camera peak
    above         = floor_peak_y[floor_peak_y > ransac_y]
    ground_peak_y = above[0] if len(above) > 0 \
                    else floor_peak_y[np.argmin(np.abs(floor_peak_y - ransac_y))]
    camera_height = float(ground_peak_y - ransac_y)
    ground_idx    = int(np.searchsorted(floor_peak_y, ground_peak_y))
    # Boundaries at floor surfaces: peak_i - camera_height
    bounds = np.concatenate([[y_min_glob - 1.0],
                             floor_peak_y[1:] - camera_height,
                             [y_max_glob + 1.0]])
    print(f"    RANSAC floor Y = {ransac_y:.3f} m")
    print(f"    Camera height above floor = {camera_height:.3f} m")
    print(f"    Floor 0 = peak index {ground_idx} (Y={ground_peak_y:.3f} m)")
else:
    camera_height = None
    ground_idx    = 0
    bounds = np.concatenate([[y_min_glob - 1.0],
                             [0.5 * (floor_peak_y[i] + floor_peak_y[i + 1])
                              for i in range(n_floors - 1)],
                             [y_max_glob + 1.0]])

# ---------------------------------------------------------------------------
# 4. Segment per floor, pre-compute color arrays
# ---------------------------------------------------------------------------
class FloorData:
    __slots__ = ("floor_n", "lo", "hi", "y_center", "peak_y",
                 "pts",            # (N, 3) float32, Y centered around slab midpoint
                 "colors_orig",    # (N, 3) uint8
                 "colors_height",  # (N, 3) uint8 plasma by Y
                 "colors_floor",   # (N, 3) uint8 solid floor tint
                 "cam_entries",    # list[dict]
                 "cam_centers_c",  # (M, 3) float32, Y-centered
                 )

floor_list: list[FloorData] = []

print(f"\n[4] Segmenting {n_floors} floor(s) …")
for i in range(n_floors):
    lo, hi  = bounds[i], bounds[i + 1]
    floor_n = i - ground_idx
    fd      = FloorData()
    fd.floor_n  = floor_n
    fd.lo, fd.hi = float(lo), float(hi)
    fd.peak_y    = float(floor_peak_y[i])
    fd.y_center  = float(0.5 * (lo + hi))

    mask_pcd = (pcd_y >= lo) & (pcd_y <= hi)
    pts_f    = pts_all[mask_pcd]
    rgb_f    = rgb_all[mask_pcd]

    pts_c = pts_f.copy()
    pts_c[:, 1] -= fd.y_center        # center Y around 0
    fd.pts           = pts_c
    fd.colors_orig   = rgb_f
    fd.colors_height = height_colors_u8(pts_f[:, 1])
    solid = _FLOOR_PALETTE[i % len(_FLOOR_PALETTE)]
    fd.colors_floor  = np.broadcast_to(solid, (len(pts_c), 3)).astype(np.uint8)

    mask_cam = (cam_y >= lo) & (cam_y <= hi)
    fd.cam_entries   = [entries[j] for j in np.where(mask_cam)[0]]
    cc = cam_centers[mask_cam].copy()
    cc[:, 1] -= fd.y_center
    fd.cam_centers_c = cc

    print(f"    Floor {floor_n:+d}  Y∈[{lo:.2f},{hi:.2f}]  "
          f"{mask_pcd.sum():,} pts  |  {len(fd.cam_entries)} cams")
    floor_list.append(fd)

# ---------------------------------------------------------------------------
# 5. Start viser server
# ---------------------------------------------------------------------------
server = viser.ViserServer(port=args.port)
print(f"\nViser running at http://localhost:{args.port}")
print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} "
      f"-J <user>@euler.ethz.ch <user>@<compute-node>\n")

# ---------------------------------------------------------------------------
# 6. GUI
# ---------------------------------------------------------------------------
with server.gui.add_folder("Scene"):
    info_lines = [
        f"**Scene:** {args.scene_id}",
        f"**Floors:** {n_floors}  |  **Cameras:** {len(entries)}",
    ]
    if ransac_y is not None:
        info_lines.append(f"**RANSAC floor Y:** {ransac_y:.3f} m")
    server.gui.add_markdown("  \n".join(info_lines))

with server.gui.add_folder("Controls"):
    gui_sep    = server.gui.add_slider("Separation (m)",
                                       min=0.0, max=25.0, step=0.5,
                                       initial_value=args.init_sep)
    gui_ptsize = server.gui.add_slider("Point size",
                                       min=0.003, max=0.06, step=0.001,
                                       initial_value=0.012)
    gui_color  = server.gui.add_dropdown("Color mode",
                                         options=["Original", "By height", "By floor"],
                                         initial_value="Original")

gui_visible: list = []
for i, fd in enumerate(floor_list):
    label = ("Floor 0 (ground)" if fd.floor_n == 0
             else f"Floor {fd.floor_n:+d}")
    with server.gui.add_folder(label):
        server.gui.add_markdown(
            f"Peak Y: **{fd.peak_y:.2f} m**  |  "
            f"Slab: [{fd.lo:.2f}, {fd.hi:.2f}] m  |  "
            f"**{len(fd.cam_entries)} cameras**"
        )
        gui_visible.append(server.gui.add_checkbox("Visible", initial_value=True))

with server.gui.add_folder("Selected frame"):
    gui_frame_info = server.gui.add_html("<i>Click a camera frustum</i>")
    gui_kf_image   = server.gui.add_image(
        np.zeros((270, 480, 3), dtype=np.uint8), label=None, format="jpeg")

# ---------------------------------------------------------------------------
# 7. Scene state  (module-level so callbacks can mutate)
# ---------------------------------------------------------------------------
_pcd_handles: list = [None] * n_floors
_frus_handles: list[tuple] = []    # (handle, floor_idx, pos_centered_xyz_tuple)

def _get_colors(fd: FloorData, mode: str) -> np.ndarray:
    if mode == "By height": return fd.colors_height
    if mode == "By floor":  return fd.colors_floor
    return fd.colors_orig

def _sep_pos(floor_idx: int) -> tuple[float, float, float]:
    return (0.0, floor_idx * gui_sep.value, 0.0)

# ---------------------------------------------------------------------------
# 8. Build initial scene
# ---------------------------------------------------------------------------
def build_scene():
    global _frus_handles
    _frus_handles = []
    mode = gui_color.value

    for i, fd in enumerate(floor_list):
        _pcd_handles[i] = server.scene.add_point_cloud(
            name     = f"/floors/floor_{fd.floor_n}",
            points   = fd.pts,
            colors   = _get_colors(fd, mode),
            point_size = gui_ptsize.value,
            position = _sep_pos(i),
        )

        floor_color = tuple(int(x) for x in _FLOOR_PALETTE[i % len(_FLOOR_PALETTE)])
        cam_info = cameras_bin.get(fd.cam_entries[0]["cam_id"], {}) if fd.cam_entries else {}
        fov    = cam_info.get("fov",    float(np.deg2rad(60)))
        aspect = cam_info.get("aspect", 4 / 3)

        for j, entry in enumerate(fd.cam_entries):
            qw, qx, qy, qz = entry["qvec"]
            wxyz_wc = (qw, -qx, -qy, -qz)   # conjugate: R_cw → R_wc
            cc      = fd.cam_centers_c[j]
            cc_tup  = (float(cc[0]), float(cc[1]), float(cc[2]))
            world_y = cc_tup[1] + i * gui_sep.value

            fh = server.scene.add_camera_frustum(
                name     = f"/cams/floor_{fd.floor_n}/{entry['name']}",
                fov      = fov,
                aspect   = aspect,
                scale    = 0.07,
                color    = floor_color,
                wxyz     = wxyz_wc,
                position = (cc_tup[0], world_y, cc_tup[2]),
            )
            _frus_handles.append((fh, i, cc_tup))

            def _make_click(name, floor_label, cc_tup_inner):
                def _on_click(_):
                    img_path = KF_DIR / name
                    gui_frame_info.content = (
                        f"<b>{name}</b><br>"
                        f"Floor: <b>{floor_label}</b><br>"
                        f"Cam center (centered): "
                        f"({cc_tup_inner[0]:.2f}, {cc_tup_inner[1]:.2f}, {cc_tup_inner[2]:.2f})"
                    )
                    if img_path.exists():
                        gui_kf_image.image = load_image_array(img_path)
                return _on_click

            lbl = "0 (ground)" if fd.floor_n == 0 else f"{fd.floor_n:+d}"
            fh.on_click(_make_click(entry["name"], lbl, cc_tup))

print("Building scene …")
build_scene()
print(f"  {n_floors} floor point clouds + "
      f"{len(_frus_handles)} camera frustums added.")

# ---------------------------------------------------------------------------
# 9. GUI callbacks
# ---------------------------------------------------------------------------
@gui_sep.on_update
def _on_sep(_):
    """Update positions only — no re-upload."""
    with server.atomic():
        for i, h in enumerate(_pcd_handles):
            if h is not None:
                h.position = _sep_pos(i)
        for (fh, fi, cc) in _frus_handles:
            fh.position = (cc[0], cc[1] + fi * gui_sep.value, cc[2])


@gui_ptsize.on_update
def _on_ptsize(_):
    with server.atomic():
        for h in _pcd_handles:
            if h is not None:
                h.point_size = gui_ptsize.value


@gui_color.on_update
def _on_color(_):
    """Re-upload colors (positions preserved via explicit position arg)."""
    mode = gui_color.value
    with server.atomic():
        for i, fd in enumerate(floor_list):
            h = server.scene.add_point_cloud(
                name       = f"/floors/floor_{fd.floor_n}",
                points     = fd.pts,
                colors     = _get_colors(fd, mode),
                point_size = gui_ptsize.value,
                position   = _sep_pos(i),
            )
            _pcd_handles[i] = h


for _i, _vis_cb in enumerate(gui_visible):
    def _make_vis_cb(floor_idx):
        def _on_vis(_):
            vis = gui_visible[floor_idx].value
            h = _pcd_handles[floor_idx]
            if h is not None:
                h.visible = vis
            for (fh, fi, _) in _frus_handles:
                if fi == floor_idx:
                    fh.visible = vis
        return _on_vis
    _vis_cb.on_update(_make_vis_cb(_i))

# ---------------------------------------------------------------------------
# Keep alive
# ---------------------------------------------------------------------------
print("\nViewer ready. Press Ctrl+C to stop.\n")
try:
    server.sleep_forever()
except KeyboardInterrupt:
    print("Shutting down.")
