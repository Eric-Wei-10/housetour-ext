"""
Oriented bounding-box utilities for XZ density histograms.

Shared by viewer.py (online / double-click) and export_room_bbox.py (offline batch).

Two-stage pipeline:
  1. detect_dominant_angle  — Radon-like projection on thresholded wall skeleton
  2. find_bbox_at_angle     — sliding-window search at the detected orientation
"""

import numpy as np


def detect_dominant_angle(hist: np.ndarray, x_edges: np.ndarray, z_edges: np.ndarray,
                          density_percentile: float = 80,
                          n_probe: int = 180) -> tuple[float, np.ndarray]:
    """Detect the dominant wall orientation from high-density bins.

    1. Threshold the histogram to keep only the top (100 - density_percentile)%
       density bins → a sparse "wall skeleton".
    2. For each of *n_probe* angles in [0, pi), project the skeleton bin centres
       onto that direction and measure the L2 norm (sum-of-squares) of the 1-D
       projection histogram.  When the projection direction is perpendicular to
       a wall, the wall collapses into a narrow spike → high L2.
    3. The angle with the highest L2 score is the wall *normal*.  The bbox edges
       align with both the normal direction and the wall direction (normal + 90°).

    Returns (dominant_theta, skeleton_mask).
    """
    nz_vals = hist[hist > 0]
    if len(nz_vals) < 10:
        return 0.0, np.zeros_like(hist, dtype=bool)

    threshold = np.percentile(nz_vals, density_percentile)
    skeleton = hist >= threshold

    sk_rows, sk_cols = np.nonzero(skeleton)
    if len(sk_rows) < 4:
        return 0.0, skeleton

    x_c = 0.5 * (x_edges[sk_rows] + x_edges[sk_rows + 1])
    z_c = 0.5 * (z_edges[sk_cols] + z_edges[sk_cols + 1])

    x_c = x_c - x_c.mean()
    z_c = z_c - z_c.mean()

    bin_sz = min(x_edges[1] - x_edges[0], z_edges[1] - z_edges[0])
    angles = np.linspace(0, np.pi, n_probe, endpoint=False)
    scores = np.zeros(n_probe)

    for i, phi in enumerate(angles):
        proj = x_c * np.cos(phi) + z_c * np.sin(phi)
        p_range = proj.max() - proj.min()
        if p_range < 1e-6:
            continue
        n_bins = max(10, int(p_range / bin_sz))
        h, _ = np.histogram(proj, bins=n_bins)
        scores[i] = (h ** 2).sum()

    dominant_theta = float(angles[scores.argmax()])
    return dominant_theta, skeleton


def find_bbox_at_angle(hist: np.ndarray, x_edges: np.ndarray, z_edges: np.ndarray,
                       com_xz: np.ndarray, theta: float,
                       coverage: float = 0.95) -> tuple[np.ndarray, float] | None:
    """Minimum-area bbox at fixed orientation *theta*, covering *coverage* of weight.

    Each non-zero histogram bin is a weighted 2D point (weight = bin count).
    Rotates into an axis-aligned frame at angle *theta*, then runs a two-level
    sliding-window search:
      1. Outer: sweep left boundary along the primary axis (u), for each left
         boundary enumerate right boundaries from the tightest valid one outward.
      2. Inner: for bins inside the u-range, sort by secondary axis (v) and use
         vectorised searchsorted to find the narrowest v-interval whose
         cumulative weight >= target and that contains the (rotated) CoM.

    Returns (corners (4,2) in original XZ coords, area), or None.
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

    best_area = np.inf
    best_info = None

    step_l = max(1, M // 50)
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

        step_r = max(1, (M - r_start) // 30)
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
                best_info = (float(u_lo), float(u_hi),
                             float(v_sorted[s_best]), float(v_sorted[e_best]))

    if best_info is None:
        return None
    u_lo, u_hi, v_lo, v_hi = best_info
    R_inv = np.array([[ct, -st], [st, ct]])
    corners_rot = np.array([[u_lo, v_lo], [u_hi, v_lo],
                             [u_hi, v_hi], [u_lo, v_hi]])
    return corners_rot @ R_inv.T, best_area
