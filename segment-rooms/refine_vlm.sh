#!/bin/bash
#SBATCH --job-name=refine_vlm
#SBATCH --output=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/refine_vlm_%j.out
#SBATCH --error=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/refine_vlm_%j.err
#SBATCH --chdir=/cluster/project/cvg/students/xinwei/housetour-ext
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_4090:1

# Usage:
#   Single scene  : sbatch ./segment-rooms/refine_vlm.sh 14
#   Several scenes: sbatch ./segment-rooms/refine_vlm.sh 11 14 1002
#   Range         : sbatch ./segment-rooms/refine_vlm.sh 100-200
#   Mixed         : sbatch ./segment-rooms/refine_vlm.sh 11 100-200 1624
#   With options  : sbatch ./segment-rooms/refine_vlm.sh 11 14 --n_frames 12
#   Dry run       : sbatch ./segment-rooms/refine_vlm.sh 11 14 --dry_run
#
# Pass-through options (refine_vlm.py flags):
#   --n_frames    Frames sampled per segment (default: 10)
#   --max_depth   Max recursive split depth (default: 2)
#   --cache_dir   Model cache directory
#   --dry_run     Print what would change without modifying files

module load stack/2024-06
module load cuda/12.4
module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/xinwei/qwen-vl

SCRIPT_DIR="/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms"
mkdir -p "$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# Parse arguments: scene IDs (individual or START-END ranges) and optional
# flags forwarded verbatim to refine_vlm.py.
# ---------------------------------------------------------------------------
SCENE_IDS=()
SCRIPT_ARGS=()

KNOWN_FLAGS=(--n_frames --max_depth --cache_dir)
BOOL_FLAGS=(--dry_run)

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"

    is_flag=0
    for flag in "${KNOWN_FLAGS[@]}"; do
        if [[ "$arg" == "$flag" ]]; then
            SCRIPT_ARGS+=("$arg" "${args[$((i+1))]}")
            i=$((i+2))
            is_flag=1
            break
        fi
    done
    [[ $is_flag -eq 1 ]] && continue

    is_bool=0
    for flag in "${BOOL_FLAGS[@]}"; do
        if [[ "$arg" == "$flag" ]]; then
            SCRIPT_ARGS+=("$arg")
            i=$((i+1))
            is_bool=1
            break
        fi
    done
    [[ $is_bool -eq 1 ]] && continue

    if [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        START="${BASH_REMATCH[1]}"
        END="${BASH_REMATCH[2]}"
        for ((j=START; j<=END; j++)); do
            SCENE_IDS+=("$j")
        done
    else
        SCENE_IDS+=("$arg")
    fi
    i=$((i+1))
done

if [ ${#SCENE_IDS[@]} -eq 0 ]; then
    echo "Error: no scene IDs provided."
    echo ""
    echo "Usage:"
    echo "  Single scene  : sbatch ./segment-rooms/refine_vlm.sh 14"
    echo "  Several scenes: sbatch ./segment-rooms/refine_vlm.sh 11 14 1002"
    echo "  Range         : sbatch ./segment-rooms/refine_vlm.sh 100-200"
    echo "  Mixed         : sbatch ./segment-rooms/refine_vlm.sh 11 100-200 1624"
    echo "  With options  : sbatch ./segment-rooms/refine_vlm.sh 11 14 --n_frames 12"
    exit 1
fi

echo "=== refine_vlm pipeline ==="
echo "Scenes       : ${SCENE_IDS[*]}"
echo "N scenes     : ${#SCENE_IDS[@]}"
[ ${#SCRIPT_ARGS[@]} -gt 0 ] && echo "Script args  : ${SCRIPT_ARGS[*]}"
echo "Started      : $(date)"

# ---------------------------------------------------------------------------
# Run refine_vlm.py once with all scene IDs so the model is loaded once.
# ---------------------------------------------------------------------------
python "$SCRIPT_DIR/refine_vlm.py" \
    --scene_ids "${SCENE_IDS[@]}" \
    "${SCRIPT_ARGS[@]}"

RC=$?
echo ""
echo "=== Pipeline complete ==="
echo "Finished : $(date)"
if [ $RC -ne 0 ]; then
    echo "refine_vlm.py exited with code $RC"
    exit $RC
fi
