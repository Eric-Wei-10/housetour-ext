"""
Print aggregate statistics across eval_vlm.json results for a list of scene IDs.

Usage:
    python summarize_eval.py 11 14 1002
    python summarize_eval.py 100 101 102 ...
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG_ROOT = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/label_segments"

scene_ids = sys.argv[1:]
if not scene_ids:
    sys.exit(0)

EXCLUDE_LABELS = {"hallway"}

total_segs = total_match = 0
per_class_total = defaultdict(int)
per_class_match = defaultdict(int)
scene_rows = []

for sid in scene_ids:
    p = Path(SEG_ROOT) / sid / "eval_vlm.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    segs = [s for s in data["segments"] if s["siglip_label"] not in EXCLUDE_LABELS]
    n = len(segs)
    m = sum(1 for s in segs if s["match"])
    total_segs  += n
    total_match += m
    acc = m / n if n else 0.0
    scene_rows.append((sid, m, n, acc))
    for seg in segs:
        label = seg["siglip_label"]
        per_class_total[label] += 1
        if seg["match"]:
            per_class_match[label] += 1

if not scene_rows:
    print("No eval_vlm.json files found for the given scene IDs.")
    sys.exit(0)

overall = total_match / total_segs if total_segs else 0.0
accuracies = [acc for _, _, _, acc in scene_rows]
median = sorted(accuracies)[len(accuracies) // 2] if accuracies else 0.0

W = 52
print()
print("=" * W)
print(f"  SUMMARY  (excluded: {', '.join(sorted(EXCLUDE_LABELS))})")
print("=" * W)
print(f"  Scenes evaluated : {len(scene_rows)}")
print(f"  Total segments   : {total_segs}")
print(f"  Overall accuracy : {total_match}/{total_segs}  ({overall:.1%})")
print(f"  Median accuracy  : {median:.1%}")

print()
print(f"  {'Scene':<10} {'Match':>8} {'Total':>7} {'Acc':>7}")
print(f"  {'-'*10} {'-'*8} {'-'*7} {'-'*7}")
for sid, m, n, acc in sorted(scene_rows, key=lambda x: int(x[0])):
    print(f"  {sid:<10} {m:>8} {n:>7} {acc:>7.1%}")

print()
print(f"  {'Room type':<18} {'Match':>8} {'Total':>7} {'Acc':>7}")
print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*7}")
for label in sorted(per_class_total, key=lambda l: -per_class_total[l]):
    n = per_class_total[label]
    m = per_class_match[label]
    print(f"  {label:<18} {m:>8} {n:>7} {m/n:>7.1%}")
print("=" * W)

# ── Accuracy histogram ────────────────────────────────────────────────────
out_path = Path(SEG_ROOT) / "eval_accuracy_histogram.png"

fig, ax = plt.subplots(figsize=(7, 4))
import numpy as np
bins = np.arange(0.4, 1.05, 0.05)
ax.hist(accuracies, bins=bins, edgecolor="white", color="#3498db")
ax.axvline(overall, color="#e74c3c", linewidth=1.5, linestyle="--",
           label=f"mean {overall:.1%}")
ax.axvline(median, color="#e67e22", linewidth=1.5, linestyle=":",
           label=f"median {median:.1%}")
ax.set_xlim(0.4, 1.0)
ax.set_xlabel("Accuracy")
ax.set_ylabel("Frequency (scenes)")
ax.set_title(f"Per-scene accuracy  (n={len(accuracies)}, excluded: {', '.join(sorted(EXCLUDE_LABELS))})")
ax.legend()
fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\n  Histogram saved to {out_path}")
