#!/bin/bash
#SBATCH --job-name=segment_floors
#SBATCH --output=/cluster/project/cvg/students/xinwei/housetour-ext/segment-floors/logs/segment_floors_%j.out
#SBATCH --error=/cluster/project/cvg/students/xinwei/housetour-ext/segment-floors/logs/segment_floors_%j.err
#SBATCH --chdir=/cluster/project/cvg/students/xinwei/housetour-ext
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --gpus=1

# Usage:
#   Single scene  : sbatch segment-floors/segment_floors.sbatch 14
#   Several scenes: sbatch segment-floors/segment_floors.sbatch 11 14 1002
#   Range         : sbatch segment-floors/segment_floors.sbatch 100-200
#   Mixed         : sbatch segment-floors/segment_floors.sbatch 11 100-200 1624
#   With options  : sbatch segment-floors/segment_floors.sbatch 11 14 --model vitl14 --skip 5 --bin_size 0.1

module load stack/2024-06
module load cuda/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/xinwei/housetour

SCRIPT_DIR="/cluster/project/cvg/students/xinwei/housetour-ext/segment-floors"
mkdir -p "$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# Parse arguments: scene IDs (individual or START-END ranges) and optional flags.
#   SCALE_ARGS   → scale_pointcloud.py  (--model, --skip)
#   SEGMENT_ARGS → segment_floors.py    (--bin_size)
# ---------------------------------------------------------------------------
SCENE_IDS=()
SCALE_ARGS=()
SEGMENT_ARGS=()

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    if [[ "$arg" == "--model" || "$arg" == "--skip" ]]; then
        SCALE_ARGS+=("$arg" "${args[$((i+1))]}")
        i=$((i+2))
    elif [[ "$arg" == "--bin_size" ]]; then
        SEGMENT_ARGS+=("$arg" "${args[$((i+1))]}")
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
    echo "  Single scene  : sbatch segment_floors.sbatch 14"
    echo "  Several scenes: sbatch segment_floors.sbatch 11 14 1002"
    echo "  Range         : sbatch segment_floors.sbatch 100-200"
    echo "  Mixed         : sbatch segment_floors.sbatch 11 100-200 1624"
    echo "  With options  : sbatch segment_floors.sbatch 11 14 --model vitl14 --skip 5 --bin_size 0.1"
    exit 1
fi

echo "=== segment_floors pipeline ==="
echo "Scenes       : ${SCENE_IDS[*]}"
echo "N scenes     : ${#SCENE_IDS[@]}"
[ ${#SCALE_ARGS[@]}   -gt 0 ] && echo "Scale args   : ${SCALE_ARGS[*]}"
[ ${#SEGMENT_ARGS[@]} -gt 0 ] && echo "Segment args : ${SEGMENT_ARGS[*]}"
echo "Started      : $(date)"

# ---------------------------------------------------------------------------
# Run pipeline for each scene
# ---------------------------------------------------------------------------
run_scene() {
    local SCENE_ID="$1"

    local SCENE_DIR="/cluster/project/cvg/students/xinwei/official-housetour-dataset/reconstructions/${SCENE_ID}"
    if [ ! -d "$SCENE_DIR" ]; then
        return 2
    fi

    echo ""
    echo "================================================================"
    echo "Scene ${SCENE_ID}  ($(date))"
    echo "================================================================"

    echo "--- Step 1: align_scale_pointcloud.py ---"
    python "$SCRIPT_DIR/align_scale_pointcloud.py" "$SCENE_ID" "${SCALE_ARGS[@]}"
    if [ $? -ne 0 ]; then
        echo "Error: align_scale_pointcloud.py failed for scene ${SCENE_ID}."
        return 1
    fi

    echo "--- Step 2: segment_floors.py ---"
    python "$SCRIPT_DIR/segment_floors.py" "$SCENE_ID" "${SEGMENT_ARGS[@]}"
    if [ $? -ne 0 ]; then
        echo "Error: segment_floors.py failed for scene ${SCENE_ID}."
        return 1
    fi

    local PLY_ORIG="${SCENE_DIR}/${SCENE_ID}_pointcloud.ply"
    if [ -f "$PLY_ORIG" ]; then
        rm "$PLY_ORIG"
        echo "Deleted ${PLY_ORIG}"
    fi

    return 0
}

N_OK=0
N_SKIP=0
N_FAIL=0
FAILED=()

for SCENE_ID in "${SCENE_IDS[@]}"; do
    run_scene "$SCENE_ID"
    RC=$?
    if [ $RC -eq 0 ]; then
        N_OK=$((N_OK + 1))
    elif [ $RC -eq 2 ]; then
        N_SKIP=$((N_SKIP + 1))
    else
        N_FAIL=$((N_FAIL + 1))
        FAILED+=("$SCENE_ID")
    fi
done

echo ""
echo "=== Pipeline complete ==="
echo "Finished : $(date)"
echo "Success  : ${N_OK} / ${#SCENE_IDS[@]}"
echo "Skipped  : ${N_SKIP}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed   : ${FAILED[*]}"
    exit 1
fi
