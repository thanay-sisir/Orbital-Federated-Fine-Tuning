#!/usr/bin/env bash
#SBATCH --job-name=orbital-continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:5
#SBATCH --constraint=gpupod
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
: "${ORBITAL_CONFIG:?Set the epoch two or three configuration}"
: "${ORBITAL_PARENT_RUN:?Set the completed previous epoch run directory}"
srun --ntasks=1 scripts/orbital_sft run --config "$ORBITAL_CONFIG" --run-id "epoch-${SLURM_JOB_ID}" --parent-run "$ORBITAL_PARENT_RUN"
