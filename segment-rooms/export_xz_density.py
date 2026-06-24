"""
Batch-export XZ density maps, center-of-mass, and oriented bounding boxes
for all segments in a scene.

Replicates the viewer's double-click pipeline (multi-view filter → dense/sparse
point extraction → histogram → bbox) but runs offline without viser.

Usage:
    python segment-rooms/export_xz_density.py --scene_id 7
    python segment-rooms/export_xz_density.py --scene_id 7 --mode sparse
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
args = parser.parse_args()

SCENE_DIR = Path(args.data_root) / args.scene_id
SEG_DIR = Path(args.seg_root) / args.scene_id
IMAGES_BIN = SCENE_DIR / "0_metric" / "images.bin"
POINTS3D_BIN = IMAGES_BIN.parent / "points3D.bin"
OUT_DIR = SEG_DIR / "xz_density"
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


# ---------------------------------------------------------------------------
# Bbox: oriented search
# ---------------------------------------------------------------------------
def find_min_bbox(hist: np.ndarray, x_edges: np.ndarray, z_edges: np.ndarray,
                  com_xz: np.ndarray, n_angles: int = 36,
                  coverage: float = 0.95) -> tuple[np.ndarray, float] | None:
    """Minimum-area oriented bbox covering *coverage* of the density, containing com_xz.

    For each of *n_angles* orientations in [0, pi), rotates into an axis-aligned
    frame and runs a two-level sliding-window search.
    """
    nz_rows, nz_cols = np.nonzero(hist)
    if len(nz_rows) < 4:
        return None
    x_c = 0.5 * (x_edges[nz_rows] + x_edges[nz_rows + 1])
    z_c = 0.5 * (z_edges[nz_cols] + z_edges[nz_cols + 1])
    w = hist[nz_rows, nz_cols].astype(np.float64)
    pts = np.column_stack([x_c, z_c])
    total_w = w.sum()
    target_w = coverage * total_w
    M = len(pts)

    best_area = np.inf
    best_info = None

    for k in range(n_angles):
        theta = np.pi * k / n_angles
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, st], [-st, ct]])
        rot = pts @ R.T
        com_r = com_xz @ R.T
        cu, cv = float(com_r[0]), float(com_r[1])

        u_order = np.argsort(rot[:, 0])
        u_s = rot[u_order, 0]
        v_s = rot[u_order, 1]
        w_s = w[u_order]
        cum_w = np.cumsum(w_s)

        step_l = max(1, M // 30)
        for l in range(0, M, step_l):
            if u_s[l] > cu:
                break
            w_before = cum_w[l - 1] if l > 0 else 0.0
            if cum_w[-1] - w_before < target_w:
                break
            r_min_w = int(np.searchsorted(cum_w, target_w + w_before, side="left"))
            r_min_w = max(r_min_w, l)
            r_min_cu = int(np.searchsorted(u_s, cu, side="left"))
            r_start = max(r_min_w, r_min_cu)
            if r_start >= M:
                continue
            step_r = max(1, (M - r_start) // 20)
            for r in range(r_start, M, step_r):
                u_lo, u_hi = u_s[l], u_s[r]
                u_width = u_hi - u_lo
                v_win = v_s[l:r + 1]
                w_win = w_s[l:r + 1]
                v_ord = np.argsort(v_win)
                v_sorted = v_win[v_ord]
                w_sorted = w_win[v_ord]
                cum_w_v = np.cumsum(w_sorted)
                n_v = len(v_sorted)
                prev_cum = np.empty(n_v)
                prev_cum[0] = 0.0
                prev_cum[1:] = cum_w_v[:-1]
                e_min = np.searchsorted(cum_w_v, target_w + prev_cum, side="left")
                s_arr = np.arange(n_v)
                valid_e = e_min < n_v
                valid_s_cv = v_sorted <= cv
                e_clamped = np.minimum(e_min, n_v - 1)
                valid_e_cv = v_sorted[e_clamped] >= cv
                valid = valid_e & valid_s_cv & valid_e_cv
                if not valid.any():
                    continue
                v_ranges = v_sorted[e_min[valid]] - v_sorted[s_arr[valid]]
                bi = v_ranges.argmin()
                area = u_width * v_ranges[bi]
                if area < best_area:
                    best_area = area
                    s_best = int(s_arr[valid][bi])
                    e_best = int(e_min[valid][bi])
                    best_info = (theta, float(u_lo), float(u_hi),
                                 float(v_sorted[s_best]), float(v_sorted[e_best]))

    if best_info is None:
        return None
    theta, u_lo, u_hi, v_lo, v_hi = best_info
    ct, st = np.cos(theta), np.sin(theta)
    R_inv = np.array([[ct, -st], [st, ct]])
    corners_rot = np.array([[u_lo, v_lo], [u_hi, v_lo],
                             [u_hi, v_hi], [u_lo, v_hi]])
    return corners_rot @ R_inv.T, best_area


# ---------------------------------------------------------------------------
# Density export
# ---------------------------------------------------------------------------
def save_xz_density(pts: np.ndarray, segment_id: int, out_dir: Path,
                    bin_size: float = 0.05, coverage: float = 0.95):
    if len(pts) == 0:
        print(f"  [seg {segment_id}] no points, skipping")
        return
    x, z = pts[:, 0], pts[:, 2]
    x_bins = max(int(np.ceil((x.max() - x.min()) / bin_size)) + 1, 1)
    z_bins = max(int(np.ceil((z.max() - z.min()) / bin_size)) + 1, 1)
    hist, x_edges, z_edges = np.histogram2d(x, z, bins=[x_bins, z_bins])

    com_x, com_z = float(x.mean()), float(z.mean())
    com_xz = np.array([com_x, com_z])

    npz_path = out_dir / f"seg_{segment_id}_xz_density.npz"
    np.savez(npz_path, hist=hist, x_edges=x_edges, z_edges=z_edges)

    bbox_result = find_min_bbox(hist, x_edges, z_edges, com_xz, coverage=coverage)

    meta = {"center_of_mass_x": com_x, "center_of_mass_z": com_z,
            "n_points": len(pts)}
    if bbox_result is not None:
        corners, area = bbox_result
        meta["bbox_corners"] = corners.tolist()
        meta["bbox_area"] = float(area)
        print(f"  [seg {segment_id}] bbox area={area:.3f} m²")
    else:
        print(f"  [seg {segment_id}] bbox search returned None")

    json_path = out_dir / f"seg_{segment_id}_com.json"
    with open(json_path, "w") as fp:
        json.dump(meta, fp, indent=2)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(hist.T, origin="lower", aspect="equal",
                   extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                   cmap="hot", interpolation="nearest")
    ax.plot(com_x, com_z, marker="+", color="cyan", markersize=14, markeredgewidth=2)
    if bbox_result is not None:
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], "-", color="cyan", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"Segment {segment_id} — XZ density (bin={bin_size}m)")
    fig.colorbar(im, ax=ax, label="Point count")
    png_path = out_dir / f"seg_{segment_id}_xz_density.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [seg {segment_id}] saved {npz_path.name}, {png_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not SCENE_DIR.exists() or not IMAGES_BIN.exists() or not POINTS3D_BIN.exists():
        return
    seg_json = SEG_DIR / "segments.json"
    if not seg_json.exists():
        return

    print(f"Scene {args.scene_id} — mode={args.mode}")

    # Load COLMAP
    print("Loading images.bin ...")
    entries = read_images_bin(IMAGES_BIN)
    print(f"  {len(entries)} registered frames")

    print("Loading points3D.bin ...")
    points3d = read_points3d_bin(POINTS3D_BIN)
    print(f"  {len(points3d)} 3D points")

    # Per-frame sparse points (quality-filtered)
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

    # Dense PLY + voxel index (for dense mode)
    dense_pts = None
    dense_clrs = None
    vox_sorted_packed = None
    vox_sort_order = None
    if args.mode == "dense":
        ply_path = SCENE_DIR / f"{args.scene_id}_metric.ply"
        if ply_path.exists():
            import open3d as o3d
            print(f"Loading dense PLY: {ply_path.name} ...")
            pcd = o3d.io.read_point_cloud(str(ply_path))
            dense_pts = np.asarray(pcd.points, dtype=np.float64)
            dense_clrs = ((np.asarray(pcd.colors) * 255).astype(np.uint8)
                          if pcd.has_colors()
                          else np.full((len(dense_pts), 3), 180, dtype=np.uint8))
            print(f"  {len(dense_pts)} dense points, building voxel index ...")
            vox_sorted_packed, vox_sort_order = _build_voxel_index(dense_pts)
            print("  done")
        else:
            print(f"  [warn] {ply_path.name} not found, falling back to sparse")

    # Load segments
    with open(seg_json) as f:
        segments = json.load(f)
    print(f"  {len(segments)} segments\n")

    # Process each segment
    for seg in segments:
        sid = seg["segment_id"]
        frame_names = seg["frame_names"]
        valid_names = [n for n in frame_names if n in name_to_sparse]
        if not valid_names:
            print(f"  [seg {sid}] no valid frames with sparse pts, skipping")
            continue

        # Gather sparse points
        all_pts = np.concatenate([name_to_sparse[n][0] for n in valid_names])
        all_clrs = np.concatenate([name_to_sparse[n][1] for n in valid_names])
        all_pids = np.concatenate([name_to_sparse[n][2] for n in valid_names])

        # Multi-view filter
        uniq_pids, first_idx, counts = np.unique(
            all_pids, return_index=True, return_counts=True)
        multi = counts >= args.min_multi_view
        sparse_pts = all_pts[first_idx[multi]]
        sparse_clrs = all_clrs[first_idx[multi]]
        print(f"  [seg {sid}] multi-view: {multi.sum()}/{len(uniq_pids)} pts "
              f"(>={args.min_multi_view} frames)")

        if len(sparse_pts) == 0:
            print(f"  [seg {sid}] no points after multi-view filter, skipping")
            continue

        # Dense or sparse mode
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

        save_xz_density(shown_pts, sid, OUT_DIR,
                         bin_size=args.bin_size, coverage=args.coverage)

    print(f"\nDone. Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
