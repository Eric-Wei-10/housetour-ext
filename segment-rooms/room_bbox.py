"""
Export room bounding boxes using dominant wall-direction detection.

Pipeline per segment:
  1. Build XZ density histogram (same as export_xz_density.py)
  2. Threshold to top-K% density bins → wall skeleton
  3. Radon-like projection on skeleton → dominant wall angle θ
  4. Fix bbox orientation to θ, find minimum-area bbox covering 95% of weight

Usage:
    python segment-rooms/room_bbox.py --scene_id 7
    python segment-rooms/room_bbox.py --scene_id 7 --mode sparse
    python segment-rooms/room_bbox.py --scene_id 7 --density_percentile 80
"""

import argparse
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bbox_utils import detect_dominant_angle, find_bbox_at_angle

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--scene_id", required=True)
parser.add_argument("--data_root", default="/cluster/project/cvg/students/xinwei/"
                                           "official-housetour-dataset/reconstructions")
parser.add_argument("--seg_root", default="/cluster/project/cvg/students/xinwei/"
                                          "official-housetour-dataset/label_segments")
parser.add_argument("--mode", default="dense", choices=["dense", "sparse"],
                    help="dense: find dense PLY points near sparse observations; "
                         "sparse: use sparse points3D.bin observations directly")
parser.add_argument("--bin_size", type=float, default=0.05,
                    help="Histogram bin size in metres")
parser.add_argument("--coverage", type=float, default=0.95,
                    help="Fraction of points the bbox must cover")
parser.add_argument("--min_multi_view", type=int, default=3,
                    help="Keep only points seen by >= this many frames in a segment")
parser.add_argument("--density_percentile", type=float, default=80,
                    help="Percentile threshold for wall skeleton (e.g. 80 = top 20%%)")
args = parser.parse_args()

SCENE_DIR = Path(args.data_root) / args.scene_id
SEG_DIR = Path(args.seg_root) / args.scene_id
IMAGES_BIN = SCENE_DIR / "0_metric" / "images.bin"
POINTS3D_BIN = IMAGES_BIN.parent / "points3D.bin"
OUT_DIR = SEG_DIR / "room_bbox"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------
def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            point3d_ids = []
            for _ in range(n2):
                f.read(16)
                pid = struct.unpack("<q", f.read(8))[0]
                if pid >= 0:
                    point3d_ids.append(pid)
            images[iid] = dict(name=name.decode(), point3d_ids=point3d_ids)
    return sorted(images.values(), key=lambda x: x["name"])


def read_points3d_bin(path):
    points = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]
            n_track = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * n_track)
            points[pid] = (xyz, rgb, error, n_track)
    return points


# ---------------------------------------------------------------------------
# Voxel helpers (for dense mode)
# ---------------------------------------------------------------------------
_VOXEL_SIZE = 0.15


def _pack_vox(vox: np.ndarray) -> np.ndarray:
    OFFSET = 1 << 13
    x = (vox[:, 0].astype(np.int64) + OFFSET) & 0x3FFF
    y = (vox[:, 1].astype(np.int64) + OFFSET) & 0x3FFF
    z = (vox[:, 2].astype(np.int64) + OFFSET) & 0x3FFF
    return x | (y << 14) | (z << 28)


def _build_voxel_index(pts: np.ndarray):
    vox_idx = np.floor(pts / _VOXEL_SIZE).astype(np.int32)
    packed = _pack_vox(vox_idx)
    order = np.argsort(packed, kind="stable")
    return packed[order], order


def _dense_near_sparse(sparse_pts, vox_sorted_packed, vox_sort_order):
    sp_vox = np.floor(sparse_pts / _VOXEL_SIZE).astype(np.int32)
    offsets = np.array([[dx, dy, dz]
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)], dtype=np.int32)
    query_vox = (sp_vox[:, None] + offsets[None]).reshape(-1, 3)
    query_keys = np.unique(_pack_vox(query_vox))
    n_vox = len(vox_sorted_packed)
    lo = np.searchsorted(vox_sorted_packed, query_keys)
    hi = np.searchsorted(vox_sorted_packed, query_keys, side="right")
    hit = hi > lo
    pieces = [vox_sort_order[lo[i]:hi[i]] for i in np.where(hit)[0]]
    if not pieces:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(pieces))


# ---------------------------------------------------------------------------
# Spatial clustering (for sparse mode)
# ---------------------------------------------------------------------------
_CLUSTER_RADIUS = 0.15


def _largest_cluster_mask(pts: np.ndarray, radius: float) -> np.ndarray:
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    vox = np.floor(pts / radius).astype(np.int32)
    uniq_vox, point_vox = np.unique(vox, axis=0, return_inverse=True)
    n_vox = len(uniq_vox)
    packed = _pack_vox(uniq_vox)
    order = np.argsort(packed)
    sorted_packed = packed[order]
    offsets = np.array([[dx, dy, dz]
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)
                        if (dx, dy, dz) != (0, 0, 0)], dtype=np.int32)
    edges_a, edges_b = [], []
    for off in offsets:
        neighbor_packed = _pack_vox(uniq_vox + off)
        pos = np.searchsorted(sorted_packed, neighbor_packed)
        valid = (pos < n_vox) & (sorted_packed[np.minimum(pos, n_vox - 1)] == neighbor_packed)
        edges_a.append(np.arange(n_vox)[valid])
        edges_b.append(order[pos[valid]])
    graph = coo_matrix((np.ones(sum(len(a) for a in edges_a), dtype=np.int8),
                        (np.concatenate(edges_a), np.concatenate(edges_b))),
                       shape=(n_vox, n_vox))
    _, vox_labels = connected_components(graph, directed=False)
    largest = np.bincount(vox_labels).argmax()
    return vox_labels[point_vox] == largest


def save_room_bbox(pts: np.ndarray, segment_id: int, out_dir: Path,
                   bin_size: float = 0.05, coverage: float = 0.95,
                   density_percentile: float = 80):
    if len(pts) == 0:
        print(f"  [seg {segment_id}] no points, skipping")
        return
    x, z = pts[:, 0], pts[:, 2]
    x_bins = max(int(np.ceil((x.max() - x.min()) / bin_size)) + 1, 1)
    z_bins = max(int(np.ceil((z.max() - z.min()) / bin_size)) + 1, 1)
    hist, x_edges, z_edges = np.histogram2d(x, z, bins=[x_bins, z_bins])

    com_x, com_z = float(x.mean()), float(z.mean())
    com_xz = np.array([com_x, com_z])

    # Step 1: detect dominant angle from wall skeleton
    dominant_theta, skeleton = detect_dominant_angle(
        hist, x_edges, z_edges, density_percentile=density_percentile)
    wall_dir_deg = float(np.rad2deg((dominant_theta + np.pi / 2) % np.pi))
    print(f"  [seg {segment_id}] dominant wall direction: {wall_dir_deg:.1f}°"
          f"  (skeleton bins: {int(skeleton.sum())}/{int((hist > 0).sum())})")

    # Step 2: find bbox at that orientation
    bbox_result = find_bbox_at_angle(hist, x_edges, z_edges, com_xz,
                                     dominant_theta, coverage=coverage)

    meta = {"center_of_mass_x": com_x, "center_of_mass_z": com_z,
            "n_points": len(pts),
            "dominant_angle_rad": float(dominant_theta),
            "wall_direction_deg": wall_dir_deg,
            "density_percentile": density_percentile}
    if bbox_result is not None:
        corners, area = bbox_result
        meta["bbox_corners"] = corners.tolist()
        meta["bbox_area"] = float(area)
        print(f"  [seg {segment_id}] bbox area={area:.3f} m²")
    else:
        corners = None
        print(f"  [seg {segment_id}] bbox search returned None")

    json_path = out_dir / f"seg_{segment_id}.json"
    with open(json_path, "w") as fp:
        json.dump(meta, fp, indent=2)

    # --- Visualisation: 2-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: density map + bbox
    ax = axes[0]
    im = ax.imshow(hist.T, origin="lower", aspect="equal",
                   extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                   cmap="hot", interpolation="nearest")
    ax.plot(com_x, com_z, marker="+", color="cyan", markersize=14, markeredgewidth=2)
    if corners is not None:
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], "-", color="cyan", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"Seg {segment_id} — density + bbox")
    fig.colorbar(im, ax=ax, label="Point count")

    # Right: wall skeleton + dominant direction line
    ax = axes[1]
    im2 = ax.imshow(skeleton.T.astype(float), origin="lower", aspect="equal",
                    extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                    cmap="gray", interpolation="nearest")
    wall_dx = np.cos(dominant_theta + np.pi / 2)
    wall_dz = np.sin(dominant_theta + np.pi / 2)
    line_len = max(x_edges[-1] - x_edges[0], z_edges[-1] - z_edges[0]) * 0.6
    ax.plot([com_x - wall_dx * line_len, com_x + wall_dx * line_len],
            [com_z - wall_dz * line_len, com_z + wall_dz * line_len],
            "-", color="cyan", linewidth=2, label=f"wall dir {wall_dir_deg:.0f}°")
    ax.plot(com_x, com_z, marker="+", color="red", markersize=14, markeredgewidth=2)
    ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"Seg {segment_id} — skeleton (top {100 - density_percentile:.0f}%)")
    fig.colorbar(im2, ax=ax, label="Skeleton")

    fig.tight_layout()
    png_path = out_dir / f"seg_{segment_id}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [seg {segment_id}] saved {json_path.name}, {png_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not SCENE_DIR.exists() or not IMAGES_BIN.exists() or not POINTS3D_BIN.exists():
        return
    seg_json = SEG_DIR / "segments.json"
    if not seg_json.exists():
        return

    print(f"Scene {args.scene_id} — mode={args.mode}, "
          f"density_percentile={args.density_percentile}")

    print("Loading images.bin ...")
    entries = read_images_bin(IMAGES_BIN)
    print(f"  {len(entries)} registered frames")

    print("Loading points3D.bin ...")
    points3d = read_points3d_bin(POINTS3D_BIN)
    print(f"  {len(points3d)} 3D points")

    name_to_sparse: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for entry in entries:
        ids = entry.get("point3d_ids") or []
        valid_pids = [pid for pid in ids
                      if pid in points3d
                      and points3d[pid][2] < 1.0
                      and points3d[pid][3] >= 3]
        if valid_pids:
            valid = [points3d[pid] for pid in valid_pids]
            name_to_sparse[entry["name"]] = (
                np.array([v[0] for v in valid], dtype=np.float64),
                np.array([v[1] for v in valid], dtype=np.uint8),
                np.array(valid_pids, dtype=np.int64),
            )
    print(f"  {len(name_to_sparse)} frames have sparse observations")

    dense_pts = None
    vox_sorted_packed = None
    vox_sort_order = None
    if args.mode == "dense":
        ply_path = SCENE_DIR / f"{args.scene_id}_metric.ply"
        if ply_path.exists():
            import open3d as o3d
            print(f"Loading dense PLY: {ply_path.name} ...")
            pcd = o3d.io.read_point_cloud(str(ply_path))
            dense_pts = np.asarray(pcd.points, dtype=np.float64)
            print(f"  {len(dense_pts)} dense points, building voxel index ...")
            vox_sorted_packed, vox_sort_order = _build_voxel_index(dense_pts)
            print("  done")
        else:
            print(f"  [warn] {ply_path.name} not found, falling back to sparse")

    with open(seg_json) as f:
        segments = json.load(f)
    print(f"  {len(segments)} segments\n")

    for seg in segments:
        sid = seg["segment_id"]
        frame_names = seg["frame_names"]
        valid_names = [n for n in frame_names if n in name_to_sparse]
        if not valid_names:
            print(f"  [seg {sid}] no valid frames, skipping")
            continue

        all_pts = np.concatenate([name_to_sparse[n][0] for n in valid_names])
        all_pids = np.concatenate([name_to_sparse[n][2] for n in valid_names])

        uniq_pids, first_idx, counts = np.unique(
            all_pids, return_index=True, return_counts=True)
        multi = counts >= args.min_multi_view
        sparse_pts = all_pts[first_idx[multi]]
        print(f"  [seg {sid}] multi-view: {multi.sum()}/{len(uniq_pids)} pts")

        if len(sparse_pts) == 0:
            continue

        if args.mode == "dense" and vox_sorted_packed is not None:
            dense_idx = _dense_near_sparse(sparse_pts, vox_sorted_packed, vox_sort_order)
            print(f"  [seg {sid}] dense: {len(sparse_pts)} sparse → {len(dense_idx)} dense")
            if len(dense_idx) == 0:
                continue
            shown_pts = dense_pts[dense_idx]
        elif args.mode == "sparse":
            mask = _largest_cluster_mask(sparse_pts, _CLUSTER_RADIUS)
            print(f"  [seg {sid}] cluster: {mask.sum()}/{len(mask)} in largest")
            shown_pts = sparse_pts[mask]
        else:
            shown_pts = sparse_pts

        save_room_bbox(shown_pts, sid, OUT_DIR,
                       bin_size=args.bin_size, coverage=args.coverage,
                       density_percentile=args.density_percentile)

    print(f"\nDone. Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
