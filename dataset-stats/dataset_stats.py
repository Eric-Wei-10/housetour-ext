"""
Survey all scenes in the reconstructions directory and report statistics
on how many COLMAP sub-reconstructions (subdirectories named with integers:
0/, 1/, ...) each scene has.

Outputs:
  - reconstruction_stats.txt  (text table + distribution summary)
  - reconstruction_hist.png   (histogram of sub-reconstruction counts)
"""

import argparse
import os
import struct
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_scene_range(s):
    """Parse '1-277' or '7,42,100' or mixed '1-10,20,30-40' into a set of ints."""
    ids = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids

parser = argparse.ArgumentParser(description="COLMAP reconstruction statistics")
parser.add_argument("--scenes", default=None,
                    help="Scene filter: range or list, e.g. '1-277' or '7,42' or '1-10,20'")
args = parser.parse_args()

RECON_ROOT = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_n_images(images_bin_path):
    """Return number of registered camera poses in a COLMAP images.bin."""
    try:
        with open(images_bin_path, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    except Exception:
        return 0


def get_sub_reconstructions(scene_dir):
    """Return sorted list of (part_name, n_cameras) for all integer-named sub-dirs."""
    subs = []
    try:
        for e in sorted(os.scandir(scene_dir), key=lambda e: int(e.name)
                        if e.name.isdigit() else float("inf")):
            if not (e.is_dir() and e.name.isdigit()):
                continue
            images_bin = os.path.join(e.path, "images.bin")
            n = read_n_images(images_bin) if os.path.exists(images_bin) else 0
            subs.append((e.name, n))
    except PermissionError:
        pass
    return subs


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
allowed = parse_scene_range(args.scenes) if args.scenes else None

scene_ids = sorted(
    (e.name for e in os.scandir(RECON_ROOT)
     if e.is_dir() and not e.name.startswith(".")
     and (allowed is None or (e.name.isdigit() and int(e.name) in allowed))),
    key=lambda x: int(x) if x.isdigit() else x
)

# scene_data: {scene_id: [(part, n_cameras), ...]}
scene_data = {sid: get_sub_reconstructions(os.path.join(RECON_ROOT, sid))
              for sid in scene_ids}

n_subs = {sid: len(v) for sid, v in scene_data.items()}
vals   = list(n_subs.values())
freq   = Counter(vals)

# Scenes with more than one sub-reconstruction
multi = {sid: v for sid, v in scene_data.items() if len(v) > 1}

# ---------------------------------------------------------------------------
# Build text output
# ---------------------------------------------------------------------------
lines = []

# --- Full table ---
lines.append(f"{'Scene':<10} {'#Parts':>6}  {'Cameras per part'}")
lines.append("-" * 60)
for sid in scene_ids:
    subs = scene_data[sid]
    parts_str = "  ".join(f"{name}:{n}" for name, n in subs) if subs else "(none)"
    lines.append(f"{sid:<10} {len(subs):>6}  {parts_str}")

lines.append("")

# --- Multi-part detail table ---
lines.append("=" * 60)
lines.append(f"Scenes with multiple sub-reconstructions ({len(multi)} total):")
lines.append("=" * 60)
lines.append(f"{'Scene':<10} {'Part':<6} {'#Cameras':>10}")
lines.append("-" * 30)
for sid in scene_ids:
    subs = scene_data[sid]
    if len(subs) <= 1:
        continue
    for part, n_cams in subs:
        lines.append(f"{sid:<10} {part:<6} {n_cams:>10}")
    lines.append(f"{'':10} {'total':<6} {sum(n for _, n in subs):>10}")
    lines.append("")

# --- Summary ---
lines.append("Distribution of #sub-reconstructions:")
for k in sorted(freq):
    lines.append(f"  {k} part(s): {freq[k]} scene(s)")
lines.append("")
lines.append(f"Total scenes        : {len(scene_ids)}")
lines.append(f"Multi-part scenes   : {len(multi)}")
lines.append(f"Min parts per scene : {min(vals)}")
lines.append(f"Max parts per scene : {max(vals)}")
lines.append(f"Mean parts per scene: {np.mean(vals):.2f}")

text = "\n".join(lines)
print(text)

txt_path = os.path.join(SCRIPT_DIR, "reconstruction_stats.txt")
with open(txt_path, "w") as f:
    f.write(text + "\n")
print(f"\nStats saved  → {txt_path}")

# ---------------------------------------------------------------------------
# Plot 1: distribution histogram (unchanged)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("COLMAP Sub-Reconstruction Count Distribution", fontsize=14, fontweight="bold")

bins = np.arange(0.5, max(vals) + 1.5, 1)
ax.hist(vals, bins=bins, color="#3498db", edgecolor="white", rwidth=0.8)
ax.set_xlabel("#Sub-reconstructions")
ax.set_ylabel("Number of scenes")
ax.set_xticks(range(1, max(vals) + 1))
for k, v in freq.items():
    ax.text(k, v + 0.1, str(v), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
png_path = os.path.join(SCRIPT_DIR, "reconstruction_hist.png")
plt.savefig(png_path, dpi=150)
print(f"Plot saved   → {png_path}")

# ---------------------------------------------------------------------------
# Plot 2: stacked bar — cameras per part for multi-part scenes
# ---------------------------------------------------------------------------
if multi:
    max_parts = max(len(v) for v in multi.values())
    part_colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12",
                   "#9B59B6", "#1ABC9C", "#E67E22"]

    multi_ids = [sid for sid in scene_ids if sid in multi]
    x = np.arange(len(multi_ids))
    width = 0.6

    fig2, ax2 = plt.subplots(figsize=(max(10, len(multi_ids) * 0.5), 5))
    fig2.suptitle("Camera poses per part — multi-part scenes",
                  fontsize=13, fontweight="bold")

    bottoms = np.zeros(len(multi_ids))
    for part_idx in range(max_parts):
        part_vals = []
        for sid in multi_ids:
            subs = multi[sid]
            part_vals.append(subs[part_idx][1] if part_idx < len(subs) else 0)
        part_vals = np.array(part_vals)
        bars = ax2.bar(x, part_vals, width, bottom=bottoms,
                       color=part_colors[part_idx % len(part_colors)],
                       label=f"part {part_idx}")
        # Annotate non-zero segments
        for xi, (val, bot) in enumerate(zip(part_vals, bottoms)):
            if val > 0:
                ax2.text(xi, bot + val / 2, str(val),
                         ha="center", va="center", fontsize=7, color="white",
                         fontweight="bold")
        bottoms += part_vals

    ax2.set_xticks(x)
    ax2.set_xticklabels(multi_ids, rotation=90, fontsize=8)
    ax2.set_ylabel("Number of camera poses")
    ax2.set_xlabel("Scene ID")
    ax2.legend(title="Part", fontsize=8, loc="upper right")
    plt.tight_layout()

    png2_path = os.path.join(SCRIPT_DIR, "multipart_cameras.png")
    fig2.savefig(png2_path, dpi=150)
    print(f"Plot saved   → {png2_path}")
