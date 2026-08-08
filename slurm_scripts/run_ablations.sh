#!/bin/bash
# Launches the 3 ablations (no_warmup, no_wd, single_noise) in parallel.
# Each job trains and evaluates automatically.
# Run from the repository root: ./slurm_scripts/run_ablations.sh

set -e

CONFIGS=(
    "configs/ablation_no_warmup.yaml"
    "configs/ablation_no_wd.yaml"
    "configs/ablation_single_noise.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo "Submitting ablation: $cfg"
    sbatch --export=ALL,ABL_CONFIG="$cfg" slurm_scripts/ablation.srm
done

echo
echo "All ablation jobs submitted. Check with: squeue -u $USER"
