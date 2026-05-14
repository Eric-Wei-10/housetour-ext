#!/bin/bash
#SBATCH --job-name=evaluate_vlm
#SBATCH --output=/cluster/project/cvg/students/xinwei/housetour-ext/eval/logs/evaluate_vlm_%j.out
#SBATCH --error=/cluster/project/cvg/students/xinwei/housetour-ext/eval/logs/evaluate_vlm_%j.err
#SBATCH --chdir=/cluster/project/cvg/students/xinwei/housetour-ext
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --gpus=rtx_4090:1

# Usage:
#   Single scene  : sbatch ./evaluate_vlm.sh 14
#   Several scenes: sbatch ./evaluate_vlm.sh 11 14 1002
#   Range         : sbatch ./evaluate_vlm.sh 100-200
#   Mixed         : sbatch ./evaluate_vlm.sh 11 100-200 1624
#   Stats only    : sbatch ./evaluate_vlm.sh --stats-only 100-200

module load stack/2024-06
module load cuda/12.4
module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/xinwei/qwen-vl

SCRIPT_DIR="/cluster/project/cvg/students/xinwei/housetour-ext/eval"

# ---------------------------------------------------------------------------
# Parse arguments: optional --stats-only flag, then scene IDs / ranges.
# ---------------------------------------------------------------------------
SCENE_IDS=()
STATS_ONLY=0

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    if [[ "$arg" == "--stats-only" ]]; then
        STATS_ONLY=1
    elif [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
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
    echo "  Single scene  : sbatch ./evaluate_vlm.sh 14"
    echo "  Several scenes: sbatch ./evaluate_vlm.sh 11 14 1002"
    echo "  Range         : sbatch ./evaluate_vlm.sh 100-200"
    echo "  Mixed         : sbatch ./evaluate_vlm.sh 11 100-200 1624"
    echo "  Stats only    : sbatch ./evaluate_vlm.sh --stats-only 100-200"
    exit 1
fi

echo "=== evaluate_vlm ==="
echo "Scenes    : ${SCENE_IDS[*]}"
echo "N scenes  : ${#SCENE_IDS[@]}"
echo "Started   : $(date)"

# ---------------------------------------------------------------------------
# Run evaluate_vlm.py (skipped with --stats-only).
# ---------------------------------------------------------------------------
if [ $STATS_ONLY -eq 0 ]; then
    python "$SCRIPT_DIR/evaluate_vlm.py" \
        --scene_ids "${SCENE_IDS[@]}"

    RC=$?
    echo ""
    echo "=== Eval done ==="
    echo "Finished : $(date)"
    if [ $RC -ne 0 ]; then
        echo "evaluate_vlm.py exited with code $RC"
        exit $RC
    fi
fi

python "$SCRIPT_DIR/summarize_eval.py" "${SCENE_IDS[@]}"
