#!/bin/bash
#SBATCH --job-name=extract_reconstructions
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00

ml stack/2025-06 gcc/12.2.0
ml pigz/2.8

ARCHIVE="/cluster/work/cvg/data/HouseTourDataset/official-housetour-dataset.tar.gz"
DEST="/cluster/scratch/xinwei/housetour_dataset"

mkdir -p "$DEST"

echo "Listing archive to find scene IDs <= 277... (first 100 scenes)"
# Extract top-level scene directory names from the archive, keep those <= 277
PATHS=$(tar -I pigz -tf "$ARCHIVE" \
    --wildcards "official-housetour-dataset/reconstructions/*/" \
    | grep -oP 'reconstructions/\K[0-9]+(?=/)' \
    | sort -un \
    | awk '$1+0 <= 277 {print "official-housetour-dataset/reconstructions/"$1}')

echo "Found scenes: $(echo "$PATHS" | wc -w)"

echo "Extracting..."
tar -I pigz -xf "$ARCHIVE" \
    -C "$DEST" \
    $PATHS

echo "Extraction complete. Scenes extracted:"
ls "$DEST/official-housetour-dataset/reconstructions/" | wc -l
echo "Scene IDs:"
ls "$DEST/official-housetour-dataset/reconstructions/"
