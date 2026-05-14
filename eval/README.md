# eval

Evaluation pipeline for room-segment labels, using Qwen2.5-VL-7B as pseudo
ground truth.

## Scripts

### `evaluate_vlm.py` / `evaluate_vlm.sh`

For each segment, samples up to 5 frames and asks Qwen2.5-VL to classify the
room type. Repeats the inference 5 times (frames may differ across calls) and
takes a majority vote. Compares the VLM answer against the SigLIP smoothed label
and writes results to `eval_vlm.json`.

**Output** (per scene under `label_segments/<scene_id>/`):

| File | Description |
|---|---|
| `eval_vlm.json` | per-segment results: siglip label, vlm label, match flag, vote counts |
| `eval_viz/<seg_id>.png` | comparison figure showing sampled frames and label votes |

```bash
sbatch ./eval/evaluate_vlm.sh 14
sbatch ./eval/evaluate_vlm.sh 11 14 1002
sbatch ./eval/evaluate_vlm.sh 100-200
sbatch ./eval/evaluate_vlm.sh --stats-only 100-200   # skip inference, only summarize
```

### `summarize_eval.py`

Reads `eval_vlm.json` for each scene and prints aggregate accuracy statistics
(overall and per room class). Excludes `hallway` segments from scoring.
Also called automatically at the end of `evaluate_vlm.sh`.

```bash
python eval/summarize_eval.py 11 14 1002
python eval/summarize_eval.py 100-200
```
