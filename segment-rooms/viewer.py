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

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
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
parser.add_argument("--double_click_mode", default="dense",
                    choices=["dense", "sparse"],
                    help="dense: voxel-grid search for dense PLY points near the "
                         "segment's points3D.bin observations; "
                         "sparse: show the segment's points3D.bin points directly")
parser.add_argument("--max_display_pts", type=int, default=1,
                    help="Max points in initial/reset view (0 = no limit)")
args = parser.parse_args()

SCENE_DIR = Path(args.data_root) / args.scene_id
SEG_DIR   = Path(args.seg_root)  / args.scene_id
KF_DIR    = SCENE_DIR / "keyframes"
SEG_JSON   = SEG_DIR / "segments.json"
LABELS_NPY = SEG_DIR / "room_labels.npy"

IMAGES_BIN = SCENE_DIR / "0_metric" / "images.bin"
if not IMAGES_BIN.exists():
    raise FileNotFoundError(f"No 0_metric/images.bin found under {SCENE_DIR}")

POINTS3D_BIN = IMAGES_BIN.parent / "points3D.bin"

# ---------------------------------------------------------------------------
# COLMAP reader
# ---------------------------------------------------------------------------
def read_images_bin(path, read_pts2d: bool = False):
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
            if read_pts2d:
                point3d_ids = []
                for _ in range(n2):
                    f.read(16)  # x2, y2 (unused)
                    pid = struct.unpack("<q", f.read(8))[0]
                    if pid >= 0:
                        point3d_ids.append(pid)
            else:
                f.read(24 * n2)
                point3d_ids = None
            images[iid] = dict(name=name.decode(),
                               qvec=np.array([qw, qx, qy, qz]),
                               tvec=np.array([tx, ty, tz]),
                               cam_id=cam_id,
                               point3d_ids=point3d_ids)
    return sorted(images.values(), key=lambda x: x["name"])

def read_points3d_bin(path):
    """Returns {point3D_id: (xyz, rgb, error, n_track)}."""
    points = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid     = struct.unpack("<Q", f.read(8))[0]
            xyz     = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            rgb     = np.array(struct.unpack("<3B", f.read(3)),  dtype=np.uint8)
            error   = struct.unpack("<d", f.read(8))[0]
            n_track = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * n_track)
            points[pid] = (xyz, rgb, error, n_track)
    return points

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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading poses ...")
entries = read_images_bin(IMAGES_BIN, read_pts2d=True)
print(f"  {len(entries)} registered frames")

# pre-compute per-frame sparse point arrays from points3D.bin
name_to_sparse_pts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
if not POINTS3D_BIN.exists():
    print(f"  [warn] points3D.bin not found — point-cloud highlight unavailable")
else:
    print("Loading points3D.bin ...")
    points3d = read_points3d_bin(POINTS3D_BIN)
    for entry in entries:
        ids = entry.get("point3d_ids") or []
        valid_pids = [pid for pid in ids
                      if pid in points3d
                      and points3d[pid][2] < 1.0    # reprojection error < 1px
                      and points3d[pid][3] >= 3]     # observed by >= 3 cameras
        if valid_pids:
            valid = [points3d[pid] for pid in valid_pids]
            name_to_sparse_pts[entry["name"]] = (
                np.array([v[0] for v in valid], dtype=np.float64),
                np.array([v[1] for v in valid], dtype=np.uint8),
                np.array(valid_pids, dtype=np.int64),
            )
    print(f"  {len(name_to_sparse_pts)} frames have sparse point observations")

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
    gui_reset_btn   = server.gui.add_button("Reset view")

@gui_reset_btn.on_click
def _on_reset(_):
    _reset_points()

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

# double-click detection
_last_click: dict = {"name": "", "time": 0.0}
# segment_id → list of frame names that have frustums
seg_to_names: dict[int, list[str]] = {}
# handle for the selection sphere indicator
_sphere_handle = {"h": None}
# handle for the segment centroid marker (double-click)
_centroid_handle = {"h": None}

# --double_click_mode sparse: points within this radius are considered
# spatially connected when finding the largest "near" cluster.
_CLUSTER_RADIUS = 0.15

def _largest_cluster_mask(pts: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask selecting the largest spatially-connected component of pts.

    Points are grouped into a voxel grid (cell size = radius); two points
    are considered connected if their voxels are the same or 26-adjacent.
    Connected components are computed on the (much smaller) voxel graph and
    mapped back to points. The mask marks the largest component (the "near"
    cluster).
    """
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)

    vox = np.floor(pts / radius).astype(np.int32)
    uniq_vox, point_vox = np.unique(vox, axis=0, return_inverse=True)
    n_vox = len(uniq_vox)

    packed        = _pack_vox(uniq_vox)
    order         = np.argsort(packed)
    sorted_packed = packed[order]

    offsets = np.array([[dx, dy, dz]
                         for dx in (-1, 0, 1)
                         for dy in (-1, 0, 1)
                         for dz in (-1, 0, 1)
                         if (dx, dy, dz) != (0, 0, 0)], dtype=np.int32)

    edges_a, edges_b = [], []
    for off in offsets:
        neighbor_packed = _pack_vox(uniq_vox + off)
        pos   = np.searchsorted(sorted_packed, neighbor_packed)
        valid = (pos < n_vox) & (sorted_packed[np.minimum(pos, n_vox - 1)] == neighbor_packed)
        edges_a.append(np.arange(n_vox)[valid])
        edges_b.append(order[pos[valid]])

    graph = coo_matrix((np.ones(sum(len(a) for a in edges_a), dtype=np.int8),
                         (np.concatenate(edges_a), np.concatenate(edges_b))),
                        shape=(n_vox, n_vox))
    _, vox_labels = connected_components(graph, directed=False)
    largest = np.bincount(vox_labels).argmax()
    return vox_labels[point_vox] == largest

# double-click results accumulate here, keyed by segment_id, until reset
_accumulated_segments: dict[int, tuple[np.ndarray, np.ndarray]] = {}

DENSITY_OUT_DIR = SEG_DIR / "xz_density"
DENSITY_OUT_DIR.mkdir(parents=True, exist_ok=True)

def _save_xz_density(pts: np.ndarray, segment_id: int, bin_size: float = 0.05):
    """Save a 2D density histogram of pts projected onto the XZ plane."""
    if len(pts) == 0:
        return
    x, z = pts[:, 0], pts[:, 2]
    x_bins = int(np.ceil((x.max() - x.min()) / bin_size)) + 1
    z_bins = int(np.ceil((z.max() - z.min()) / bin_size)) + 1
    x_bins = max(x_bins, 1)
    z_bins = max(z_bins, 1)

    hist, x_edges, z_edges = np.histogram2d(x, z, bins=[x_bins, z_bins])

    com_x, com_z = float(x.mean()), float(z.mean())

    # Save raw numpy
    npy_path = DENSITY_OUT_DIR / f"seg_{segment_id}_xz_density.npy"
    np.save(npy_path, hist)

    com_path = DENSITY_OUT_DIR / f"seg_{segment_id}_com.json"
    with open(com_path, "w") as fp:
        json.dump({"center_of_mass_x": com_x, "center_of_mass_z": com_z,
                    "n_points": len(pts)}, fp, indent=2)

    # Save visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(hist.T, origin="lower", aspect="equal",
                   extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                   cmap="hot", interpolation="nearest")
    ax.plot(com_x, com_z, marker="+", color="cyan", markersize=14, markeredgewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"Segment {segment_id} — XZ density (bin={bin_size}m)")
    fig.colorbar(im, ax=ax, label="Point count")
    png_path = DENSITY_OUT_DIR / f"seg_{segment_id}_xz_density.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  XZ density saved: {npy_path.name}, {png_path.name}, CoM=({com_x:.3f}, {com_z:.3f})")

def _show_points_for_frames(frame_names: list[str], is_segment: bool = False,
                             segment_id: int | None = None):
    """Display 3D points for the given frame(s).

    is_segment=False (single-click): show one frame's sparse points directly.
    is_segment=True  (double-click): show the whole segment's points.
      - First filters to points observed by >=2 frames (multi-view filter).
      - Then depending on --double_click_mode:
          dense:  find nearby dense PLY points via voxel-grid lookup.
          sparse: keep only the largest spatial cluster (drop outliers).
      - Results accumulate across double-clicks until reset.
    """
    if not frame_names:
        return
    valid_names = [n for n in frame_names if n in name_to_sparse_pts]
    if not valid_names:
        return

    all_pts  = np.concatenate([name_to_sparse_pts[n][0] for n in valid_names])
    all_clrs = np.concatenate([name_to_sparse_pts[n][1] for n in valid_names])

    if is_segment:
        # Multi-view filter: keep only points seen by >= 2 frames in this segment
        all_pids = np.concatenate([name_to_sparse_pts[n][2] for n in valid_names])
        uniq_pids, first_idx, counts = np.unique(all_pids, return_index=True, return_counts=True)
        multi = counts >= 3
        all_pts  = all_pts[first_idx[multi]]
        all_clrs = all_clrs[first_idx[multi]]
        print(f"  multi-view filter: {multi.sum()}/{len(uniq_pids)} unique pts seen by >=3 frames")

    if is_segment and args.double_click_mode == "dense" \
            and pcd_loaded and _vox_sorted_packed is not None:
        # Dense mode: find dense PLY points near the filtered sparse points
        dense_idx  = _dense_near_sparse(all_pts)
        print(f"  dense: {len(all_pts)} sparse → {len(dense_idx)} dense pts")
        shown_pts  = pts_all[dense_idx]
        shown_clrs = clrs_all[dense_idx]
        point_size = 0.008
    else:
        shown_pts  = all_pts
        shown_clrs = all_clrs

        if is_segment and args.double_click_mode == "sparse":
            # Sparse mode: keep only the largest spatial cluster
            mask = _largest_cluster_mask(shown_pts, _CLUSTER_RADIUS)
            print(f"  sparse cluster: {mask.sum()}/{len(mask)} pts in largest cluster")
            shown_pts  = shown_pts[mask]
            shown_clrs = shown_clrs[mask]

        point_size = 0.02

    if is_segment and segment_id is not None:
        # Accumulate across double-clicks
        _accumulated_segments[segment_id] = (shown_pts, shown_clrs)
        display_pts  = np.concatenate([p for p, _ in _accumulated_segments.values()], axis=0)
        display_clrs = np.concatenate([c for _, c in _accumulated_segments.values()], axis=0)
        print(f"  accumulated: {len(_accumulated_segments)} segment(s), "
              f"{len(display_pts)} pts total")
    else:
        display_pts, display_clrs = shown_pts, shown_clrs

    server.scene.add_point_cloud(
        name="point_cloud",
        points=display_pts,
        colors=display_clrs,
        point_size=point_size)

    if is_segment and len(shown_pts) > 0:
        # Show centroid marker for the current segment's points
        centroid = shown_pts.mean(axis=0)
        if _centroid_handle["h"] is not None:
            _centroid_handle["h"].remove()
        _centroid_handle["h"] = server.scene.add_icosphere(
            name="/centroid", radius=0.15,
            color=(0, 200, 255), position=tuple(centroid))
        _save_xz_density(shown_pts, segment_id)

def _reset_points():
    _accumulated_segments.clear()
    if pcd_loaded:
        server.scene.add_point_cloud(
            name="point_cloud", points=_display_pts, colors=_display_clrs, point_size=0.008)
    if _sphere_handle["h"] is not None:
        _sphere_handle["h"].remove()
        _sphere_handle["h"] = None
    if _centroid_handle["h"] is not None:
        _centroid_handle["h"].remove()
        _centroid_handle["h"] = None

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
    frustum_wxyz = (qw, -qx, -qy, -qz)
    frustum_pos  = tuple(center)
    seg_to_names.setdefault(sid, []).append(entry["name"])
    handle = server.scene.add_camera_frustum(
        name     = f"/cameras/{entry['name']}",
        fov      = float(np.deg2rad(60)),
        aspect   = 4 / 3,
        scale    = 0.08,
        color    = color,
        wxyz     = frustum_wxyz,
        position = frustum_pos,
    )

    def make_click_handler(name, path, segment_id, label_idx, pos):
        def on_click(_):
            color_hex = rgb_to_hex(SEG_PALETTE[segment_id])
            gui_frame_label.content = (
                f'<span style="color:{color_hex}">&#9632;</span> '
                f'<b>{name}</b><br>'
                f'Segment: <b>seg {segment_id}</b><br>'
                f'Room: <b>{room_name(label_idx)}</b>'
            )
            gui_image.image = load_image_array(path)

            now = time.time()
            time_diff = now - _last_click["time"]
            is_double = (_last_click["name"] == name and time_diff < 0.8)
            print(f"  click {name}: diff={time_diff:.3f}s, is_double={is_double}")
            _last_click["name"] = name
            _last_click["time"] = now

            if is_double:
                # Double-click: show all points for this segment
                if _sphere_handle["h"] is not None:
                    _sphere_handle["h"].remove()
                    _sphere_handle["h"] = None
                _show_points_for_frames(seg_to_names.get(segment_id, []),
                                        is_segment=True, segment_id=segment_id)
            else:
                # Single click: show points for this frustum only
                if _sphere_handle["h"] is not None:
                    _sphere_handle["h"].remove()
                if _centroid_handle["h"] is not None:
                    _centroid_handle["h"].remove()
                    _centroid_handle["h"] = None
                _sphere_handle["h"] = server.scene.add_icosphere(
                    name="/selected", radius=0.05,
                    color=(255, 220, 0), position=pos)
                _show_points_for_frames([name])
        return on_click

    handle.on_click(make_click_handler(
        entry["name"], KF_DIR / entry["name"], sid,
        name_to_label.get(entry["name"], -1),
        frustum_pos))

print(f"Added {len(entries) // args.every_n} frustums.")

# ---------------------------------------------------------------------------
# Optional: load point cloud
# ---------------------------------------------------------------------------
# Kept at module level so frustum click handlers can reference them without recapturing.
pts_all:  np.ndarray | None = None
clrs_all: np.ndarray | None = None
pcd_loaded = False
_VOXEL_SIZE = 0.15
_vox_sorted_packed: np.ndarray | None = None  # sorted int64 voxel keys for all dense pts
_vox_sort_order:    np.ndarray | None = None  # argsort that produced _vox_sorted_packed

def _pack_vox(vox: np.ndarray) -> np.ndarray:
    """Pack (K, 3) int32 voxel indices into (K,) int64 keys (collision-free for ±8192/axis)."""
    OFFSET = 1 << 13
    x = (vox[:, 0].astype(np.int64) + OFFSET) & 0x3FFF
    y = (vox[:, 1].astype(np.int64) + OFFSET) & 0x3FFF
    z = (vox[:, 2].astype(np.int64) + OFFSET) & 0x3FFF
    return x | (y << 14) | (z << 28)

def _dense_near_sparse(sparse_pts: np.ndarray) -> np.ndarray:
    """Return indices into pts_all for dense points within one voxel of any sparse point."""
    if _vox_sorted_packed is None:
        return np.array([], dtype=np.int64)
    sp_vox = np.floor(sparse_pts / _VOXEL_SIZE).astype(np.int32)       # (K, 3)
    offsets = np.array([[dx, dy, dz]
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)], dtype=np.int32)          # (27, 3)
    query_vox  = (sp_vox[:, None] + offsets[None]).reshape(-1, 3)       # (K*27, 3)
    query_keys = np.unique(_pack_vox(query_vox))                         # sorted unique keys
    lo = np.searchsorted(_vox_sorted_packed, query_keys)
    hi = np.searchsorted(_vox_sorted_packed, query_keys, side="right")
    hit = hi > lo
    pieces = [_vox_sort_order[lo[i]:hi[i]] for i in np.where(hit)[0]]
    if not pieces:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(pieces))

ply_path = SCENE_DIR / f"{args.scene_id}_metric.ply"
if ply_path.exists():
    try:
        import open3d as o3d
        pcd  = o3d.io.read_point_cloud(str(ply_path))
        pts  = np.asarray(pcd.points, dtype=np.float64)   # (N, 3) world coords
        clrs = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() \
               else np.full((len(pts), 3), 180, dtype=np.uint8)  # (N, 3) RGB 0-255
        pts_all   = pts
        clrs_all  = clrs
        pcd_loaded = True
        vox_idx = np.floor(pts_all / _VOXEL_SIZE).astype(np.int32)
        order   = np.argsort(_pack_vox(vox_idx), kind="stable")
        _vox_sort_order    = order
        _vox_sorted_packed = _pack_vox(vox_idx)[order]
        print(f"  Voxel grid built on {len(pts_all)} points")
        # Subsample for display if exceeding cap
        if args.max_display_pts > 0 and len(pts_all) > args.max_display_pts:
            rng = np.random.default_rng(42)
            _disp_idx = np.sort(rng.choice(len(pts_all), size=args.max_display_pts, replace=False))
            _display_pts  = pts_all[_disp_idx]
            _display_clrs = clrs_all[_disp_idx]
            print(f"  Display cap: {len(pts_all)} → {args.max_display_pts} pts")
        else:
            _display_pts  = pts_all
            _display_clrs = clrs_all
        server.scene.add_point_cloud(
            name="point_cloud", points=_display_pts, colors=_display_clrs, point_size=0.008)
        print(f"Loaded point cloud: {ply_path.name} ({len(pts)} pts)")
    except Exception as e:
        print(f"Could not load point cloud: {e}")

# ---------------------------------------------------------------------------
# Keep alive
# ---------------------------------------------------------------------------
print("\nViewer ready. Press Ctrl+C to stop.\n")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down.")
