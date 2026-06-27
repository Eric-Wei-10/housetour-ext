"""
evaluate_vlm.py  —  Evaluate segment labels using Qwen2.5-VL-7B as pseudo-GT.

For each segment, samples up to 5 evenly-spaced frames and asks Qwen2.5-VL to
classify the room type.  Compares the VLM answer against the SigLIP smoothed
label and writes per-segment results to eval_vlm.json in each scene's output dir.

Usage:
    python eval_segments_vlm.py --scene_ids 7 198
    python eval_segments_vlm.py          # all scenes
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_DIR  = "/cluster/scratch/xinwei/model_checkpoints"
DATA_ROOT  = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SEG_ROOT   = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/label_segments"

HOUSE_ROOM_TYPES = [
    "living room", "bedroom", "bathroom", "kitchen", "dining room",
    "hallway", "office", "study", "garage", "laundry room",
    "storage room", "balcony", "garden", "outside",
]

PROMPT = (
    "Look at these images and identify the type of room shown. "
    "Choose exactly one from the following list and reply with only that label, nothing else:\n"
    + ", ".join(HOUSE_ROOM_TYPES)
)

N_FRAMES_PER_SEG = 5   # frames per inference call (no repeats within one call)
N_INFERENCES     = 5   # inference calls per segment (frames may repeat across calls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """Return k randomly-sampled unique indices from [0, n_frames), sorted."""
    k = min(k, n_frames)
    return sorted(random.sample(range(n_frames), k))


def classify_images(image_paths: list[Path], model, processor, device: str) -> str:
    """Ask Qwen2.5-VL to classify the room shown across multiple images."""
    images  = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=16)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()

    for label in HOUSE_ROOM_TYPES:
        if label in answer:
            return label
    return answer


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_eval_viz(seg_id, siglip_label: str, vlm_label: str, match: bool,
                  inference_votes: list, kf_dir: Path, out_path: Path) -> None:
    """Save a comparison figure for one segment.

    Layout: one row per inference call, each cell showing a thumbnail.
    Row labels show the per-inference VLM prediction.
    Title shows segment ID, SigLIP label, final VLM label, and match status.
    Border is green on match, red on mismatch.
    """
    n_rows = len(inference_votes)
    n_cols = max(len(v["frames"]) for v in inference_votes) if inference_votes else 1

    thumb_w, thumb_h = 160, 120
    label_w          = 160   # width reserved for row label on the left
    fig_w  = (label_w + n_cols * thumb_w) / 100
    fig_h  = (40 + n_rows * thumb_h) / 100   # 40px for title

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    border_color = "#2ecc71" if match else "#e74c3c"

    for r, vote in enumerate(inference_votes):
        row_label = vote["vlm_label"]
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(vote["frames"]):
                img_path = kf_dir / vote["frames"][c]
                if img_path.exists():
                    ax.imshow(Image.open(img_path).convert("RGB"))
            # Row label on the first cell
            if c == 0:
                ax.set_ylabel(f"inf {r+1}\n{row_label}",
                              fontsize=7, rotation=0, labelpad=4,
                              ha="right", va="center")
                ax.yaxis.set_label_coords(-0.02, 0.5)

    status = "✓ MATCH" if match else "✗ MISMATCH"
    fig.suptitle(
        f"Seg {seg_id}  |  SigLIP: {siglip_label}  |  VLM: {vlm_label}  |  {status}",
        fontsize=9, fontweight="bold", color=border_color,
    )

    # Light tinted background to reinforce match/mismatch
    fig.patch.set_facecolor("#eafaf1" if match else "#fdedec")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def eval_scene(scene_id: str, model, processor, device: str) -> None:
    seg_dir  = Path(SEG_ROOT)  / scene_id
    kf_dir   = Path(DATA_ROOT) / scene_id / "keyframes"
    seg_json = seg_dir / "segments.json"
    smoothed_npy = seg_dir / "room_labels_smoothed.npy"

    if not seg_json.exists():
        print(f"  [skip] no segments.json in {seg_dir}")
        return
    if not smoothed_npy.exists():
        print(f"  [skip] no room_labels_smoothed.npy in {seg_dir}")
        return
    if (seg_dir / "eval_vlm.json").exists():
        print(f"  [skip] eval_vlm.json already exists in {seg_dir}")
        return

    import numpy as np
    smoothed_labels = np.load(smoothed_npy)

    with open(seg_json) as f:
        segments = json.load(f)

    results = []
    n_match = 0

    for seg in segments:
        seg_label_idx  = int(smoothed_labels[seg["frame_indices"][0]])
        seg_label_name = HOUSE_ROOM_TYPES[seg_label_idx] \
                         if seg_label_idx < len(HOUSE_ROOM_TYPES) else "unknown"

        # N_INFERENCES calls, each with N_FRAMES_PER_SEG randomly-sampled frames
        inference_votes = []
        for inf_idx in range(N_INFERENCES):
            local_indices = sample_frame_indices(seg["n_frames"], N_FRAMES_PER_SEG)
            image_paths = []
            fnames = []
            for li in local_indices:
                fname    = seg["frame_names"][li]
                img_path = kf_dir / fname
                if img_path.exists():
                    image_paths.append(img_path)
                    fnames.append(fname)
            if not image_paths:
                continue
            vlm_label = classify_images(image_paths, model, processor, device)
            inference_votes.append({"frames": fnames, "vlm_label": vlm_label})
            print(f"    seg {seg['segment_id']:02d} | inf {inf_idx+1}/{N_INFERENCES} {fnames} → {vlm_label}  (smoothed: {seg_label_name})")

        if inference_votes:
            from collections import Counter
            counts   = Counter(v["vlm_label"] for v in inference_votes)
            top_count = counts.most_common(1)[0][1]
            winners  = [label for label, cnt in counts.items() if cnt == top_count]
            vlm_label_seg = winners[0] if len(winners) == 1 else winners
        else:
            vlm_label_seg = "unknown"

        tied = isinstance(vlm_label_seg, list)
        if tied:
            print(f"    seg {seg['segment_id']:02d} | vote → TIE {vlm_label_seg}")
        else:
            print(f"    seg {seg['segment_id']:02d} | vote → {vlm_label_seg}")

        match = (seg_label_name in vlm_label_seg) if tied else (vlm_label_seg == seg_label_name)
        if match:
            n_match += 1

        save_eval_viz(
            seg_id         = seg["segment_id"],
            siglip_label   = seg_label_name,
            vlm_label      = vlm_label_seg,
            match          = match,
            inference_votes= inference_votes,
            kf_dir         = kf_dir,
            out_path       = seg_dir / "eval_viz" / f"seg_{seg['segment_id']}.png",
        )

        results.append({
            "segment_id":      seg["segment_id"],
            "n_frames":        seg["n_frames"],
            "siglip_label":    seg_label_name,
            "vlm_label":       vlm_label_seg,
            "match":           match,
            "inference_votes": inference_votes,
        })

    accuracy = n_match / len(results) if results else 0.0
    out = {"scene_id": scene_id, "accuracy": accuracy, "segments": results}
    with open(seg_dir / "eval_vlm.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"  [ok] scene {scene_id}: {n_match}/{len(results)} match  "
          f"(accuracy={accuracy:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segment labels with Qwen2.5-VL pseudo-GT"
    )
    parser.add_argument("--scene_ids", nargs="*", default=None,
                        help="Scene IDs to evaluate (default: all)")
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--seg_root",  default=SEG_ROOT)
    args = parser.parse_args()

    seg_root = Path(args.seg_root)

    if args.scene_ids is not None:
        scene_ids = args.scene_ids
    else:
        scene_ids = sorted(p.name for p in seg_root.iterdir() if p.is_dir())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID} ...")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    for scene_id in scene_ids:
        print(f"\nScene {scene_id}")
        eval_scene(scene_id, model, processor, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
