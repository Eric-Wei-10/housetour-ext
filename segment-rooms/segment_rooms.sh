#!/bin/bash
#SBATCH --job-name=segment_rooms
#SBATCH --output=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/segment_rooms_%j.out
#SBATCH --error=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/segment_rooms_%j.err
#SBATCH --chdir=/cluster/project/cvg/students/xinwei/housetour-ext
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --gpus=1

# Usage:
#   Single scene  : sbatch ./segment-rooms/segment_rooms.sh 14
#   Several scenes: sbatch ./segment-rooms/segment_rooms.sh 11 14 1002
#   Range         : sbatch ./segment-rooms/segment_rooms.sh 100-200
#   Mixed         : sbatch ./segment-rooms/segment_rooms.sh 11 100-200 1624
#   With options  : sbatch ./segment-rooms/segment_rooms.sh 11 14 --batch_size 64
#
# Pass-through options (segment_rooms.py flags):
#   --batch_size       SigLIP inference batch size (default: 32)
#   --window_half      Half-size w of voting window 2w+1 (default: 5)
#   --min_len          Drop segments shorter than this after trimming (default: 5)
#   --score_threshold  Drop segments whose max top-1 score never exceeds this (default: 0.01)

module load stack/2024-06
module load cuda/12.4
module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/xinwei/housetour

SCRIPT_DIR="/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms"
DATA_ROOT="/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions"
OUTPUT_ROOT="/cluster/project/cvg/students/xinwei/official-housetour-dataset/label_segments"
mkdir -p "$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# Parse arguments: scene IDs (individual or START-END ranges) and optional
# flags forwarded verbatim to segment_rooms.py.
# ---------------------------------------------------------------------------
SCENE_IDS=()
SCRIPT_ARGS=()

KNOWN_FLAGS=(--batch_size --window_half --min_len --score_threshold)

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    is_flag=0
    for flag in "${KNOWN_FLAGS[@]}"; do
        if [[ "$arg" == "$flag" ]]; then
            is_flag=1
            break
        fi
    done

    if [[ $is_flag -eq 1 ]]; then
        SCRIPT_ARGS+=("$arg" "${args[$((i+1))]}")
        i=$((i+2))
    elif [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        START="${BASH_REMATCH[1]}"
        END="${BASH_REMATCH[2]}"
        for ((j=START; j<=END; j++)); do
            SCENE_IDS+=("$j")
        done
        i=$((i+1))
    else
        SCENE_IDS+=("$arg")
        i=$((i+1))
    fi
done

if [ ${#SCENE_IDS[@]} -eq 0 ]; then
    echo "Error: no scene IDs provided."
    echo ""
    echo "Usage:"
    echo "  Single scene  : sbatch ./segment-rooms/segment_rooms.sh 14"
    echo "  Several scenes: sbatch ./segment-rooms/segment_rooms.sh 11 14 1002"
    echo "  Range         : sbatch ./segment-rooms/segment_rooms.sh 100-200"
    echo "  Mixed         : sbatch ./segment-rooms/segment_rooms.sh 11 100-200 1624"
    echo "  With options  : sbatch ./segment-rooms/segment_rooms.sh 11 14 --batch_size 64"
    exit 1
fi

echo "=== segment_rooms pipeline ==="
echo "Scenes       : ${SCENE_IDS[*]}"
echo "N scenes     : ${#SCENE_IDS[@]}"
[ ${#SCRIPT_ARGS[@]} -gt 0 ] && echo "Script args  : ${SCRIPT_ARGS[*]}"
echo "Started      : $(date)"

# ---------------------------------------------------------------------------
# Run segment_rooms.py once with all scene IDs so the model is loaded once.
# ---------------------------------------------------------------------------
python "$SCRIPT_DIR/segment_rooms.py" \
    --data_root   "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --scene_ids   "${SCENE_IDS[@]}" \
    "${SCRIPT_ARGS[@]}"

RC=$?
echo ""
echo "=== Pipeline complete ==="
echo "Finished : $(date)"
if [ $RC -ne 0 ]; then
    echo "segment_rooms.py exited with code $RC"
    exit $RC
fi
