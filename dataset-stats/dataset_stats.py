"""
Survey all scenes in the reconstructions directory and report statistics
on how many COLMAP sub-reconstructions (subdirectories named with integers:
0/, 1/, ...) each scene has.

Outputs:
  - reconstruction_stats.txt  (text table + distribution summary)
  - reconstruction_hist.png   (histogram of sub-reconstruction counts)
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RECON_ROOT = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def count_sub_reconstructions(scene_dir):
    """Count subdirectories with integer names (0/, 1/, 2/, ...)."""
    try:
        return sum(
            1 for e in os.scandir(scene_dir)
            if e.is_dir() and e.name.isdigit()
        )
    except PermissionError:
        return 0


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
scene_ids = sorted(
    (e.name for e in os.scandir(RECON_ROOT)
     if e.is_dir() and not e.name.startswith(".")),
    key=lambda x: int(x) if x.isdigit() else x
)

counts = {
    sid: count_sub_reconstructions(os.path.join(RECON_ROOT, sid))
    for sid in scene_ids
}

vals = list(counts.values())
freq = Counter(vals)

# ---------------------------------------------------------------------------
# Build text output
# ---------------------------------------------------------------------------
lines = []
lines.append(f"{'Scene':<10} {'#SubRecons':>10}")
lines.append("-" * 22)
for sid in scene_ids:
    lines.append(f"{sid:<10} {counts[sid]:>10}")
lines.append("")
lines.append("Distribution:")
for k in sorted(freq):
    lines.append(f"  {k} sub-recon(s): {freq[k]} scene(s)")
lines.append("")
lines.append(f"Total scenes   : {len(scene_ids)}")
lines.append(f"Min sub-recons : {min(vals)}")
lines.append(f"Max sub-recons : {max(vals)}")
lines.append(f"Mean sub-recons: {np.mean(vals):.2f}")

text = "\n".join(lines)
print(text)

txt_path = os.path.join(SCRIPT_DIR, "reconstruction_stats.txt")
with open(txt_path, "w") as f:
    f.write(text + "\n")
print(f"\nStats saved  → {txt_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("COLMAP Sub-Reconstruction Count Distribution", fontsize=14, fontweight="bold")

bins = np.arange(0.5, max(vals) + 1.5, 1)
ax.hist(vals, bins=bins, color="#3498db", edgecolor="white", rwidth=0.8)
ax.set_xlabel("#SubRecons")
ax.set_ylabel("Number of scenes")
ax.set_xticks(range(1, max(vals) + 1))
for k, v in freq.items():
    ax.text(k, v + 0.1, str(v), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
png_path = os.path.join(SCRIPT_DIR, "reconstruction_hist.png")
plt.savefig(png_path, dpi=150)
print(f"Plot saved   → {png_path}")
