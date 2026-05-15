#!/bin/bash
# Lanza las 3 ablaciones (no_warmup, no_wd, single_noise) en paralelo.
# Cada job entrena + evalúa automáticamente.

set -e

CONFIGS=(
    "configs/ablation_no_warmup.yaml"
    "configs/ablation_no_wd.yaml"
    "configs/ablation_single_noise.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo "Submitting ablation: $cfg"
    sbatch --export=ALL,ABL_CONFIG="$cfg" ablation.srm
done

echo
echo "All ablation jobs submitted. Check with: squeue -u $USER"
