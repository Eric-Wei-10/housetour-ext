from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_SEG_COLORS = [
    "#AED6F1", "#A9DFBF", "#FAD7A0", "#D7BDE2",
    "#F9E79F", "#FADBD8", "#A3E4D7", "#CCD1D1",
]


def save_distance_plot(delta, threshold, fires, segments, output_path: Path,
                       sigma: float = 2.0):
    n = len(delta)
    xs = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(12, n // 20), 4))

    for seg in segments:
        lo = max(0, seg["start_frame"] - 1)
        hi = min(n, seg["end_frame"])
        color = _SEG_COLORS[seg["segment_id"] % len(_SEG_COLORS)]
        ax.axvspan(lo, hi, alpha=0.35, color=color, label=f"seg {seg['segment_id']}")

    ax.plot(xs, delta,     color="#2471A3", linewidth=0.9, label="cosine distance")
    ax.plot(xs, threshold, color="#E67E22", linewidth=1.0,
            linestyle="--", label=f"adaptive threshold (σ={sigma})")

    boundary_xs = np.where(fires)[0]
    ax.vlines(boundary_xs, ymin=0, ymax=delta.max() * 1.05,
              colors="#C0392B", linewidth=1.2, alpha=0.8, label="boundary")

    ax.set_xlabel("frame index")
    ax.set_ylabel("cosine distance")
    ax.set_title(f"DINOv2 inter-frame cosine distance  "
                 f"({len(segments)} segments, {len(boundary_xs)} boundaries)")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, delta.max() * 1.1)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_segment_grid(segments, image_paths, output_path: Path,
                      thumb_size=(160, 120), cols=5):
    n_segs = len(segments)
    if n_segs == 0:
        return

    label_w  = 120
    cell_w, cell_h = thumb_size
    canvas = Image.new("RGB", (label_w + cols * cell_w, n_segs * cell_h), color=(40, 40, 40))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=13)
        except OSError:
            font = ImageFont.load_default()
    except Exception:
        draw, font = None, None

    for row, seg in enumerate(segments):
        indices = seg["frame_indices"]
        pick_at = np.linspace(0, len(indices) - 1, min(cols, len(indices)), dtype=int)
        picked  = [indices[k] for k in pick_at]
        y_off   = row * cell_h

        if draw is not None:
            label = f"seg {seg['segment_id']}\n{seg['n_frames']} fr"
            draw.text((4, y_off + cell_h // 2 - 16), label, font=font,
                      fill=_SEG_COLORS[seg["segment_id"] % len(_SEG_COLORS)])

        for col, frame_idx in enumerate(picked):
            x_off = label_w + col * cell_w
            try:
                thumb = (Image.open(image_paths[frame_idx])
                         .convert("RGB")
                         .resize(thumb_size, Image.LANCZOS))
                canvas.paste(thumb, (x_off, y_off))
                if draw is not None:
                    draw.rectangle(
                        [x_off, y_off, x_off + cell_w - 1, y_off + cell_h - 1],
                        outline=(80, 80, 80), width=1,
                    )
            except Exception:
                pass

    canvas.save(output_path)


def save_similarity_matrix(dino_feats: np.ndarray, output_path: Path,
                           segments=None):
    sim = dino_feats @ dino_feats.T
    n   = len(dino_feats)

    fig, ax = plt.subplots(figsize=(max(8, n // 30), max(8, n // 30)))
    im = ax.imshow(sim, origin="upper", aspect="auto", cmap="RdYlGn",
                   vmin=-1, vmax=1, interpolation="none")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")

    if segments is not None:
        for seg in segments[1:]:
            b = seg["start_frame"] - 0.5
            ax.axhline(b, color="white", linewidth=0.8, alpha=0.9)
            ax.axvline(b, color="white", linewidth=0.8, alpha=0.9)
        for seg in segments:
            mid = (seg["start_frame"] + seg["end_frame"]) / 2
            ax.text(mid, mid, str(seg["segment_id"]),
                    ha="center", va="center", fontsize=7,
                    color="black", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))
        title = f"DINOv2 pairwise cosine similarity  ({len(segments)} segments)"
    else:
        title = "DINOv2 pairwise cosine similarity"

    ax.set_xlabel("frame index")
    ax.set_ylabel("frame index")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_room_label_plot(room_labels: np.ndarray, room_scores: np.ndarray,
                        category_names: list, output_path: Path,
                        segments=None):
    """Two-panel plot: top-1 room type color strip + top-1 SigLIP score per frame."""
    n = len(room_labels)
    top1_scores = room_scores[np.arange(n), room_labels]

    n_cats  = len(category_names)
    cmap    = plt.get_cmap("tab20", n_cats)
    colors  = [cmap(i) for i in range(n_cats)]

    fig, (ax_cat, ax_score) = plt.subplots(
        2, 1, figsize=(max(14, n // 20), 5),
        gridspec_kw={"height_ratios": [1, 4]},
    )

    # ── top panel: categorical color strip ───────────────────────────────────
    label_img = room_labels[np.newaxis, :]               # (1, N)
    ax_cat.imshow(label_img, aspect="auto", origin="upper",
                  cmap=cmap, vmin=0, vmax=n_cats - 1, interpolation="nearest")
    ax_cat.set_yticks([])
    ax_cat.set_xticks([])
    ax_cat.set_title("Top-1 room type per frame (color) + SigLIP score")

    # legend patches
    from matplotlib.patches import Patch
    present = sorted(set(room_labels.tolist()))
    handles = [Patch(color=colors[i], label=category_names[i]) for i in present]
    ax_cat.legend(handles=handles, loc="upper right", fontsize=7,
                  ncol=max(1, len(present) // 4),
                  bbox_to_anchor=(1.0, 1.0), borderaxespad=0)

    # ── bottom panel: top-1 score line ────────────────────────────────────────
    xs = np.arange(n)

    if segments is not None:
        for seg in segments:
            lo = max(0, seg["start_frame"] - 1)
            hi = min(n, seg["end_frame"])
            ax_score.axvspan(lo, hi, alpha=0.15,
                             color=_SEG_COLORS[seg["segment_id"] % len(_SEG_COLORS)])

    # color each frame point by its room label
    for cat_idx in present:
        mask = room_labels == cat_idx
        ax_score.scatter(xs[mask], top1_scores[mask],
                         color=colors[cat_idx], s=4, alpha=0.7, linewidths=0,
                         label=category_names[cat_idx])

    ax_score.plot(xs, top1_scores, color="gray", linewidth=0.6, alpha=0.5, zorder=0)

    if segments is not None:
        for seg in segments[1:]:
            ax_score.axvline(seg["start_frame"] - 1, color="#C0392B",
                             linewidth=0.8, alpha=0.7)

    ax_score.set_xlabel("frame index")
    ax_score.set_ylabel("SigLIP sigmoid score (top-1)")
    ax_score.set_xlim(0, n - 1)
    ax_score.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_segment_frames(segments, image_paths, output_dir: Path,
                        room_labels=None, room_scores=None, category_names=None,
                        smoothed_labels=None):
    """Save per-segment frames.

    Without room_labels: creates symlinks to originals.
    With room_labels + room_scores: saves annotated copies with top-2 room labels.
    If smoothed_labels is provided, appends the smoothed label below the top-2 lines.
    """
    frames_root = output_dir / "frames"
    frames_root.mkdir(exist_ok=True)

    annotate = room_labels is not None and room_scores is not None

    if annotate:
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=16)
        except OSError:
            font = ImageFont.load_default()

    for seg in segments:
        seg_dir = frames_root / f"seg_{seg['segment_id']:02d}"
        seg_dir.mkdir(exist_ok=True)

        for frame_idx in seg["frame_indices"]:
            src = image_paths[frame_idx].resolve()
            dst = seg_dir / src.name

            if dst.exists() or dst.is_symlink():
                dst.unlink()

            if not annotate:
                dst.symlink_to(src)
                continue

            # Draw top-2 room labels + scores onto a copy
            img  = Image.open(src).convert("RGB")
            draw = ImageDraw.Draw(img)

            top2 = np.argsort(room_scores[frame_idx])[::-1][:2]
            lines = [
                f"#{i+1} {category_names[top2[i]]} ({room_scores[frame_idx][top2[i]]:.3f})"
                for i in range(2)
            ]
            if smoothed_labels is not None:
                smoothed_name = category_names[smoothed_labels[frame_idx]]
                lines.append(f"Smoothed: {smoothed_name}")
            text = "\n".join(lines)

            # Semi-transparent background box
            x, y, pad = 6, 6, 4
            bbox = draw.multiline_textbbox((x, y), text, font=font)
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                fill=(0, 0, 0, 160),
            )
            draw.multiline_text((x, y), text, font=font, fill=(255, 255, 255))

            img.save(dst)
