"""
refine_vlm.py — VLM-based post-segmentation for under-segmented bedrooms.

For each segment classified as bedroom:
  1. Filter ALL frames of the segment to those where a bed is clearly visible.
  2. Evenly sample min(n_frames, #bed_frames) of them, preserving temporal order.
  3. Ask Qwen2.5-VL whether all sampled frames show the SAME bed
     (prior: each bedroom has exactly one unique bed).
  4. If different, ask the VLM at which sampled frame the bed first changes.
  5. Split the segment at the detected boundary and recurse.

Modifies segments.json in-place. New sub-segment IDs are derived from the
original ID (e.g. segment 5 → 5_0, 5_1).

Usage:
    python refine_vlm.py --scene_ids 14
    python refine_vlm.py --scene_ids 11 14 1002 --n_frames 10
    python refine_vlm.py --scene_ids 1-200 --dry_run
"""

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_DIR = "/cluster/scratch/xinwei/model_checkpoints"
DATA_ROOT = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
SEG_ROOT  = "/cluster/project/cvg/students/xinwei/official-housetour-dataset/label_segments"

HOUSE_ROOM_TYPES = [
    "living room", "bedroom", "bathroom", "kitchen", "dining room",
    "hallway", "office", "study", "garage", "laundry room",
    "storage room", "balcony", "garden", "empty room",
]

N_FRAMES          = 10  # frames sampled per segment after bed filtering
MAX_DEPTH         = 2   # max recursive split depth
FILTER_BATCH_SIZE = 1   # images per VLM call during bed filtering

FILTER_PROMPT = (
    "Does this image clearly show a bed?\n"
    "Answer with exactly one word: yes or no."
)

DETECT_PROMPT = (
    "The following {n} images are frames in chronological order from a bedroom "
    "house-tour video, each showing a bed.\n"
    "Each bedroom contains exactly one unique bed. "
    "Are ALL these images showing the SAME bed?\n"
    "Answer with exactly one word: same or different."
)

LOCATE_PROMPT = (
    "The following {n} images are frames (numbered 1 to {n}) from a bedroom "
    "house-tour video, each showing a bed.\n"
    "These frames span two different bedrooms, each with a different bed. "
    "At which frame number does the bed FIRST change to a different bed?\n"
    "Answer with only the frame number (an integer between 1 and {n})."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_evenly(n: int, k: int) -> list[int]:
    """Return up to k evenly-spaced indices covering [0, n)."""
    k = min(k, n)
    if n <= k:
        return list(range(n))
    step = n / k
    return [int((i + 0.5) * step) for i in range(k)]


def run_vlm(image_paths: list[Path], prompt: str,
            model, processor, device: str,
            max_new_tokens: int = 64) -> str:
    images  = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()


def filter_frames_with_bed(
    paths: list[Path],
    local_indices: list[int],
    model, processor, device: str,
    batch_size: int = FILTER_BATCH_SIZE,
) -> tuple[list[Path], list[int]]:
    """Return (filtered_paths, filtered_local_indices) for frames where a bed is visible.

    Processes frames in batches of batch_size for better per-image accuracy.
    """
    if not paths:
        return [], []

    keep = set()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        answer = run_vlm(batch_paths, FILTER_PROMPT, model, processor, device,
                         max_new_tokens=4)
        print(f"    bed-filter [{start+1}] {paths[start].name} → '{answer}'")
        if "yes" in answer:
            keep.add(start)

    filtered_paths  = [paths[i]        for i in sorted(keep)]
    filtered_locals = [local_indices[i] for i in sorted(keep)]
    return filtered_paths, filtered_locals


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def try_split(seg: dict, kf_dir: Path,
              model, processor, device: str,
              n_frames: int, max_depth: int,
              depth: int = 0) -> list[dict]:
    """Recursively try to split a segment using the VLM. Returns a list of segment dicts."""
    if depth >= max_depth or seg["n_frames"] < 4:
        return [seg]

    # Build (path, local_index) pairs for every frame in the segment
    all_paths, all_valid = [], []
    for li in range(seg["n_frames"]):
        p = kf_dir / seg["frame_names"][li]
        if p.exists():
            all_paths.append(p)
            all_valid.append(li)

    # Filter to frames where a bed is clearly visible
    bed_paths, bed_valid = filter_frames_with_bed(
        all_paths, all_valid, model, processor, device)

    if len(bed_paths) < 2:
        print(f"    depth={depth} too few bed frames, skipping")
        return [seg]

    # Evenly sample min(n_frames, len(bed_paths)) frames, preserving temporal order
    k = min(n_frames, len(bed_paths))
    chosen = sample_evenly(len(bed_paths), k)
    sampled_paths = [bed_paths[i] for i in chosen]
    sampled_valid = [bed_valid[i]  for i in chosen]

    sampled_names = [seg["frame_names"][li] for li in sampled_valid]
    print(f"    depth={depth} bed frames: {len(bed_paths)}, sampled: {k} → {sampled_names}")

    # Step 1 — detect: are all sampled frames showing the same bed?
    detect_answer = run_vlm(
        sampled_paths,
        DETECT_PROMPT.format(n=k),
        model, processor, device,
    )
    print(f"    depth={depth} detect → '{detect_answer}'")

    if "different" not in detect_answer:
        return [seg]

    # Step 2 — locate: find where the bed first changes (reuse the same sampled frames)
    locate_answer = run_vlm(
        sampled_paths,
        LOCATE_PROMPT.format(n=k),
        model, processor, device,
    )
    print(f"    depth={depth} locate → '{locate_answer}'")

    nums = re.findall(r"\d+", locate_answer)
    if not nums:
        print(f"    could not parse boundary from '{locate_answer}', skipping split")
        return [seg]

    sampled_boundary = max(1, min(int(nums[0]), k))

    if not (0.25 * k <= sampled_boundary <= 0.75 * k):
        print(f"    depth={depth} locate boundary {sampled_boundary} outside [0.25k, 0.75k] "
              f"= [{0.25*k:.1f}, {0.75*k:.1f}], skipping split")
        return [seg]

    # sampled_boundary is the first sampled frame of the new bed (1-indexed).
    # Split at the midpoint between it and the preceding sampled frame.
    b = sampled_valid[sampled_boundary - 1]  # first local index of new bed
    if sampled_boundary == 1:
        split_pos = b  # no preceding sample; fall back to the locate frame itself
    else:
        a = sampled_valid[sampled_boundary - 2]  # last local index of old bed
        split_pos = (a + b) // 2 + 1

    if split_pos <= 0 or split_pos >= seg["n_frames"]:
        print(f"    boundary at edge (split_pos={split_pos}), skipping split")
        return [seg]

    orig_id = seg["segment_id"]

    seg_a = {
        "segment_id":    f"{orig_id}_0",
        "start_frame":   seg["start_frame"],
        "end_frame":     seg["frame_indices"][split_pos - 1],
        "n_frames":      split_pos,
        "frame_indices": seg["frame_indices"][:split_pos],
        "frame_names":   seg["frame_names"][:split_pos],
    }
    seg_b = {
        "segment_id":    f"{orig_id}_1",
        "start_frame":   seg["frame_indices"][split_pos],
        "end_frame":     seg["end_frame"],
        "n_frames":      seg["n_frames"] - split_pos,
        "frame_indices": seg["frame_indices"][split_pos:],
        "frame_names":   seg["frame_names"][split_pos:],
    }

    print(f"    split seg {orig_id} at local pos {split_pos} "
          f"({seg['frame_names'][split_pos - 1]} | {seg['frame_names'][split_pos]}) "
          f"[locate={seg['frame_names'][b]}, prev={seg['frame_names'][a] if sampled_boundary > 1 else 'n/a'}] "
          f"→ {seg_a['segment_id']} ({seg_a['n_frames']} frames), "
          f"{seg_b['segment_id']} ({seg_b['n_frames']} frames)")

    return (try_split(seg_a, kf_dir, model, processor, device, n_frames, max_depth, depth + 1) +
            try_split(seg_b, kf_dir, model, processor, device, n_frames, max_depth, depth + 1))


# ---------------------------------------------------------------------------
# Per-scene
# ---------------------------------------------------------------------------

def process_scene(scene_id: str,
                  model, processor, device: str,
                  n_frames: int, max_depth: int,
                  dry_run: bool) -> None:
    import numpy as np

    seg_dir      = Path(SEG_ROOT)  / scene_id
    kf_dir       = Path(DATA_ROOT) / scene_id / "keyframes"
    seg_json     = seg_dir / "segments.json"
    smoothed_npy = seg_dir / "room_labels_smoothed.npy"

    if not seg_json.exists():
        print(f"  [skip] no segments.json in {seg_dir}")
        return
    if not smoothed_npy.exists():
        print(f"  [skip] no room_labels_smoothed.npy in {seg_dir}")
        return

    smoothed_labels = np.load(smoothed_npy)
    segments        = json.loads(seg_json.read_text())

    new_segments = []
    n_split = 0

    for seg in segments:
        label_idx  = int(smoothed_labels[seg["frame_indices"][0]])
        label_name = HOUSE_ROOM_TYPES[label_idx] if label_idx < len(HOUSE_ROOM_TYPES) else "unknown"

        if label_name != "bedroom":
            new_segments.append(seg)
            continue

        print(f"  seg {seg['segment_id']} ({label_name}, {seg['n_frames']} frames)")
        parts = try_split(seg, kf_dir, model, processor, device, n_frames, max_depth)
        new_segments.extend(parts)
        if len(parts) > 1:
            n_split += 1

    print(f"  → {len(segments)} segments, {n_split} split → {len(new_segments)} total")

    if not dry_run:
        seg_json.write_text(json.dumps(new_segments, indent=2))
        print(f"  segments.json updated.")
    else:
        print(f"  [dry run] no files modified.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VLM-based post-segmentation for under-segmented bedrooms"
    )
    parser.add_argument("--scene_ids", required=True, nargs="+", metavar="SCENE_ID")
    parser.add_argument("--n_frames",  type=int, default=N_FRAMES,
                        help=f"Frames sampled per segment (default: {N_FRAMES})")
    parser.add_argument("--max_depth", type=int, default=MAX_DEPTH,
                        help=f"Max recursive split depth (default: {MAX_DEPTH})")
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print what would change without modifying files")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID} ...")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    # Expand ranges like "1-200" into individual IDs
    scene_ids = []
    for token in args.scene_ids:
        m = re.fullmatch(r"(\d+)-(\d+)", token)
        if m:
            scene_ids.extend(str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            scene_ids.append(token)

    print(f"Scenes: {scene_ids}  (n={len(scene_ids)})")
    for scene_id in scene_ids:
        print(f"\n--- Scene {scene_id} ---")
        process_scene(scene_id, model, processor, device,
                      args.n_frames, args.max_depth, args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
