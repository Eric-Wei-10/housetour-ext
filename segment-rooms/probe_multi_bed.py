"""
probe_multi_bed.py — Per-frame bed-count probe for bedroom segments.

For each frame in bedroom segments of a scene, asks Qwen2.5-VL:
  Q1. How many beds are visible? (none / one / multiple)
  Q2. If multiple: are they in the same room or different rooms?

Then smooths the per-frame labels and recursively detects transition points
where different beds appear, splitting the segment accordingly.

Usage:
    python probe_multi_bed.py --scene_ids 7
    python probe_multi_bed.py --scene_ids 11 14 1002
    python probe_multi_bed.py --scene_ids 1-200 --out_dir /tmp/probe
    python probe_multi_bed.py --scene_ids 7 --dry_run   # log only, no JSON splits
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
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

MIN_SEG_FRAMES = 5  # minimum frames to keep a post-split sub-segment

PROMPT_BED_COUNT = (
    "Look at this image carefully.\n"
    "How many beds are visible?\n"
    "Answer with exactly one word: none, one, or multiple."
)

PROMPT_MULTI_BED_ROOM = (
    "This image shows multiple beds.\n"
    "Are all the beds in this image in the SAME room, "
    "or do they appear to be in DIFFERENT rooms (e.g. seen through a doorway)?\n"
    "Answer with exactly one word: same or different."
)


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------
def run_vlm(image_paths: list[Path] | Path, prompt: str,
            model, processor, device: str) -> str:
    if isinstance(image_paths, Path):
        image_paths = [image_paths]
    images  = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=8)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()


def classify_frame(img_path: Path, model, processor, device: str) -> dict:
    """Run per-frame bed-count inference. Returns {frame, bed_count, same_room}."""
    bed_count = run_vlm(img_path, PROMPT_BED_COUNT, model, processor, device)
    same_room = (
        run_vlm(img_path, PROMPT_MULTI_BED_ROOM, model, processor, device)
        if "multiple" in bed_count else "n/a"
    )
    return {"frame": img_path.name, "bed_count": bed_count, "same_room": same_room}


# ---------------------------------------------------------------------------
# Label smoothing
# ---------------------------------------------------------------------------
def smooth_bed_labels(labels: list[str]) -> list[str]:
    """Two-pass smoothing on per-frame bed-count labels.

    Pass 1: isolated 'multiple' (no adjacent 'multiple' neighbor) → 'one'
    Pass 2: sliding window of length 5, weighted voting (multiple=2, others=1);
            ties broken as 'one'. Each pass reads from a fixed input list so
            no element's result affects another element's window.
    """
    n = len(labels)
    if n == 0:
        return []

    # Pass 1: remove isolated 'multiple'
    denoised = list(labels)
    for i in range(n):
        if denoised[i] == "multiple":
            has_multi_neighbor = (
                (i > 0     and denoised[i - 1] == "multiple") or
                (i < n - 1 and denoised[i + 1] == "multiple")
            )
            if not has_multi_neighbor:
                denoised[i] = "one"

    # Pass 2: weighted sliding-window voting
    WEIGHT = {"multiple": 2}  # all other labels default to 1
    smoothed = []
    for i in range(n):
        window = denoised[max(0, i - 2) : i + 3]
        scores: dict[str, int] = {}
        for label in window:
            scores[label] = scores.get(label, 0) + WEIGHT.get(label, 1)
        max_score = max(scores.values())
        winners   = [lb for lb, s in scores.items() if s == max_score]
        smoothed.append(winners[0] if len(winners) == 1 else "one")

    return smoothed


# ---------------------------------------------------------------------------
# Recursive transition splitting
# ---------------------------------------------------------------------------
def split_recursive(
    frames: list[dict],
    bed_labels: list[str],
    depth: int = 0,
) -> list[tuple[list[dict], list[str]]]:
    """Recursively split frames on 'multiple-bed' transition runs.

    Smoothing is done once before the first call; sub-segments receive slices
    of the original smoothed list — no re-smoothing on recursion.

    For each consecutive run of smoothed 'multiple':
      - Only raw-'multiple' frames cast votes (same_room answer).
      - 'different' must strictly outvote 'same' to trigger a split.
      - Transition frames (raw 'multiple' in the run) are dropped.
      - Each remaining half must have >= MIN_SEG_FRAMES to be kept.

    Returns a flat list of (frames, bed_labels) pairs for each kept sub-segment.
    """
    indent = "    " + "  " * depth
    n = len(frames)
    i = 0

    while i < n:
        if bed_labels[i] != "multiple":
            i += 1
            continue

        run_start = i
        while i < n and bed_labels[i] == "multiple":
            i += 1
        run_end = i - 1

        # Transition frames: only those that were raw 'multiple'
        transition_idxs = [
            j for j in range(run_start, run_end + 1)
            if frames[j]["bed_count"] == "multiple"
        ]
        if not transition_idxs:
            continue

        n_same = sum(1 for j in transition_idxs if frames[j]["same_room"] == "same")
        n_diff = sum(1 for j in transition_idxs if frames[j]["same_room"] == "different")

        if n_diff <= n_same:
            print(f"{indent}run [{run_start}–{run_end}]: same={n_same} diff={n_diff} → no split")
            continue

        split_start = transition_idxs[0]
        split_end   = transition_idxs[-1]
        before_frames = frames[:split_start]
        after_frames  = frames[split_end + 1:]
        before_labels = bed_labels[:split_start]
        after_labels  = bed_labels[split_end + 1:]

        print(f"{indent}run [{run_start}–{run_end}]: same={n_same} diff={n_diff} → SPLIT")
        print(f"{indent}  transition: {[frames[j]['frame'] for j in transition_idxs]}")
        print(f"{indent}  before: {len(before_frames)} frames | after: {len(after_frames)} frames")

        result: list[tuple[list[dict], list[str]]] = []
        if len(before_frames) >= MIN_SEG_FRAMES:
            result.extend(split_recursive(before_frames, before_labels, depth + 1))
        else:
            print(f"{indent}  [DROP] before: {len(before_frames)} frames < {MIN_SEG_FRAMES}")

        if len(after_frames) >= MIN_SEG_FRAMES:
            result.extend(split_recursive(after_frames, after_labels, depth + 1))
        else:
            print(f"{indent}  [DROP] after: {len(after_frames)} frames < {MIN_SEG_FRAMES}")

        return result

    return [(frames, bed_labels)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_bedroom(seg: dict, room_labels: np.ndarray) -> bool:
    label_idx = int(room_labels[seg["frame_indices"][0]])
    label     = HOUSE_ROOM_TYPES[label_idx] if label_idx < len(HOUSE_ROOM_TYPES) else "unknown"
    return label == "bedroom"


def print_frame_table(frames: list[dict],
                      bed_labels_raw: list[str],
                      bed_labels_smoothed: list[str]) -> None:
    print(f"    {'frame':<25}  {'raw':<8}  {'smoothed':<8}  rooms")
    print(f"    {'-'*25}  {'-'*8}  {'-'*8}  -----")
    for frame, raw, smoothed in zip(frames, bed_labels_raw, bed_labels_smoothed):
        flag = ""
        if smoothed == "multiple":
            flag = "  *** MULTIPLE BEDS ***"
            flag += " — DIFFERENT ROOMS" if frame["same_room"] == "different" else " — same room"
        changed = " <" if smoothed != raw else ""
        print(f"    {frame['frame']:<25}  {raw:<8}  {smoothed:<8}  {frame['same_room']}{changed}{flag}")


# ---------------------------------------------------------------------------
# Per-scene
# ---------------------------------------------------------------------------
def probe_scene(scene_id: str, model, processor, device: str,
                out_dir: Path | None, dry_run: bool = False) -> None:
    seg_dir    = Path(SEG_ROOT)  / scene_id
    kf_dir     = Path(DATA_ROOT) / scene_id / "keyframes"
    seg_json   = seg_dir / "segments.json"
    labels_npy = seg_dir / "room_labels_smoothed.npy"

    if not seg_json.exists():
        print(f"  [skip] no segments.json in {seg_dir}")
        return
    if not labels_npy.exists():
        print(f"  [skip] no room_labels_smoothed.npy in {seg_dir}")
        return

    segments    = json.loads(seg_json.read_text())
    room_labels = np.load(labels_npy)

    bedroom_segs = [seg for seg in segments if is_bedroom(seg, room_labels)]
    print(f"Scene {scene_id}: {len(bedroom_segs)} bedroom segment(s)")

    scene_results = []

    for seg in bedroom_segs:
        sid = seg["segment_id"]
        print(f"\n  === Segment {sid} ({seg['n_frames']} frames) ===")

        # Per-frame VLM inference
        frames: list[dict] = []
        for frame_name in seg["frame_names"]:
            img_path = kf_dir / frame_name
            if img_path.exists():
                frames.append(classify_frame(img_path, model, processor, device))

        # Smooth bed-count labels
        bed_labels_raw      = [f["bed_count"] for f in frames]
        bed_labels_smoothed = smooth_bed_labels(bed_labels_raw)
        for frame, smoothed in zip(frames, bed_labels_smoothed):
            frame["bed_count_smoothed"] = smoothed

        print_frame_table(frames, bed_labels_raw, bed_labels_smoothed)

        # Recursive transition analysis
        print(f"\n  --- transition analysis ---")
        sub_segs = split_recursive(frames, bed_labels_smoothed)
        print(f"  → {len(sub_segs)} sub-segment(s):")
        for k, (ss_frames, _) in enumerate(sub_segs):
            print(f"     [{k}] {ss_frames[0]['frame']} … {ss_frames[-1]['frame']}  ({len(ss_frames)} frames)")

        seg_result: dict = {"segment_id": sid, "frames": frames}
        if not dry_run:
            seg_result["sub_segments"] = [
                {"frames": [f["frame"] for f in ss_frames], "n_frames": len(ss_frames)}
                for ss_frames, _ in sub_segs
            ]
        scene_results.append(seg_result)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"probe_{scene_id}.json"
        out_path.write_text(json.dumps(scene_results, indent=2))
        print(f"\n  Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_ids", required=True, nargs="+", metavar="SCENE_ID")
    parser.add_argument("--out_dir",   default=None,
                        help="Directory for per-scene JSON output")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Run full analysis and print logs; skip writing splits to JSON")
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    args = parser.parse_args()

    # Expand ranges like "1-200"
    scene_ids: list[str] = []
    for token in args.scene_ids:
        m = re.fullmatch(r"(\d+)-(\d+)", token)
        if m:
            scene_ids.extend(str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            scene_ids.append(token)

    out_dir = Path(args.out_dir) if args.out_dir else None

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
    print(f"Model loaded. Scenes: {scene_ids}  (n={len(scene_ids)})\n")

    for scene_id in scene_ids:
        print(f"\n{'='*50}")
        print(f"Scene {scene_id}")
        print(f"{'='*50}")
        probe_scene(scene_id, model, processor, device, out_dir, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
