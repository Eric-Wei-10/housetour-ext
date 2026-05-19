#!/bin/bash
#SBATCH --job-name=probe_multi_bed
#SBATCH --output=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/probe_multi_bed_%j.out
#SBATCH --error=/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms/logs/probe_multi_bed_%j.err
#SBATCH --chdir=/cluster/project/cvg/students/xinwei/housetour-ext
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --gpus=rtx_4090:1

# Usage:
#   Single scene  : sbatch ./segment-rooms/probe_multi_bed.sh 14
#   Several scenes: sbatch ./segment-rooms/probe_multi_bed.sh 11 14 1002
#   Range         : sbatch ./segment-rooms/probe_multi_bed.sh 100-200
#   Mixed         : sbatch ./segment-rooms/probe_multi_bed.sh 11 100-200 1624
#   Log only      : sbatch ./segment-rooms/probe_multi_bed.sh 7 --dry_run
#   With output   : sbatch ./segment-rooms/probe_multi_bed.sh 7 --out_dir /tmp/probe
#
# Pass-through options (probe_multi_bed.py flags):
#   --out_dir   Directory to save per-scene probe_<id>.json results
#   --dry_run   Run full analysis and print logs; skip writing splits to JSON
#   --cache_dir Model cache directory

module load stack/2024-06
module load cuda/12.4
module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/xinwei/qwen-vl

SCRIPT_DIR="/cluster/project/cvg/students/xinwei/housetour-ext/segment-rooms"
mkdir -p "$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# Parse arguments: scene IDs (individual or START-END ranges) and optional
# flags forwarded verbatim to probe_multi_bed.py.
# ---------------------------------------------------------------------------
SCENE_IDS=()
SCRIPT_ARGS=()

KNOWN_FLAGS=(--out_dir --cache_dir)
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
    echo "  Single scene  : sbatch ./segment-rooms/probe_multi_bed.sh 14"
    echo "  Several scenes: sbatch ./segment-rooms/probe_multi_bed.sh 11 14 1002"
    echo "  Range         : sbatch ./segment-rooms/probe_multi_bed.sh 100-200"
    echo "  Mixed         : sbatch ./segment-rooms/probe_multi_bed.sh 11 100-200 1624"
    echo "  Log only      : sbatch ./segment-rooms/probe_multi_bed.sh 7 --dry_run"
    echo "  With output   : sbatch ./segment-rooms/probe_multi_bed.sh 7 --out_dir /tmp/probe"
    exit 1
fi

echo "=== probe_multi_bed ==="
echo "Scenes       : ${SCENE_IDS[*]}"
echo "N scenes     : ${#SCENE_IDS[@]}"
[ ${#SCRIPT_ARGS[@]} -gt 0 ] && echo "Script args  : ${SCRIPT_ARGS[*]}"
echo "Started      : $(date)"

python "$SCRIPT_DIR/probe_multi_bed.py" \
    --scene_ids "${SCENE_IDS[@]}" \
    "${SCRIPT_ARGS[@]}"

RC=$?
echo ""
echo "=== Done ==="
echo "Finished : $(date)"
if [ $RC -ne 0 ]; then
    echo "probe_multi_bed.py exited with code $RC"
    exit $RC
fi
