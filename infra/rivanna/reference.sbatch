#!/usr/bin/env bash
#SBATCH --job-name=orbital-five-gpu
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
config="${ORBITAL_CONFIG:-configs/orbital_sft/five_gpu.yaml}"
run_id="${ORBITAL_RUN_ID:-reference-${SLURM_JOB_ID}}"
srun --ntasks=1 scripts/orbital_sft run --config "$config" --run-id "$run_id"
