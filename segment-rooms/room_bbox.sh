#!/bin/bash
#SBATCH --job-name=room_bbox
#SBATCH --output=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/room_bbox_%j.out
#SBATCH --error=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/room_bbox_%j.err
#SBATCH --chdir=/cluster/project/cvg/students/xinwei/housetour-ext
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00

# Usage:
#   Single scene  : sbatch ./segment-rooms/room_bbox.sh 7
#   Several scenes: sbatch ./segment-rooms/room_bbox.sh 7 14 1002
#   Range         : sbatch ./segment-rooms/room_bbox.sh 100-200
#   Mixed         : sbatch ./segment-rooms/room_bbox.sh 7 100-200 1624
#   With options  : sbatch ./segment-rooms/room_bbox.sh 7 14 --mode sparse --density_percentile 70

module load stack/2024-06
module load cuda/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/xinwei/housetour

SCRIPT_DIR="/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms"
mkdir -p "$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# Parse arguments: scene IDs (individual or START-END ranges) + optional flags
# ---------------------------------------------------------------------------
SCENE_IDS=()
SCRIPT_ARGS=()

KNOWN_FLAGS=(--mode --bin_size --coverage --min_multi_view --density_percentile --data_root --seg_root)

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
    echo "  Single scene  : sbatch ./segment-rooms/room_bbox.sh 7"
    echo "  Several scenes: sbatch ./segment-rooms/room_bbox.sh 7 14 1002"
    echo "  Range         : sbatch ./segment-rooms/room_bbox.sh 100-200"
    echo "  Mixed         : sbatch ./segment-rooms/room_bbox.sh 7 100-200 1624"
    echo "  With options  : sbatch ./segment-rooms/room_bbox.sh 7 14 --mode sparse"
    exit 1
fi

echo "=== room_bbox ==="
echo "Scenes   : ${SCENE_IDS[*]}"
echo "N scenes : ${#SCENE_IDS[@]}"
[ ${#SCRIPT_ARGS[@]} -gt 0 ] && echo "Options  : ${SCRIPT_ARGS[*]}"
echo "Started  : $(date)"
echo ""

FAIL=0
for SID in "${SCENE_IDS[@]}"; do
    echo "--- scene $SID ---"
    python "$SCRIPT_DIR/room_bbox.py" \
        --scene_id "$SID" \
        "${SCRIPT_ARGS[@]}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[ERROR] scene $SID exited with code $RC"
        FAIL=$((FAIL+1))
    fi
    echo ""
done

echo "=== Done ==="
echo "Finished : $(date)"
echo "Failed   : $FAIL / ${#SCENE_IDS[@]}"
exit $FAIL
