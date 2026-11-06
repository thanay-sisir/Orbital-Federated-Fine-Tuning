#!/usr/bin/env bash
#SBATCH --job-name=orbital-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpupod
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
config="${ORBITAL_CONFIG:-configs/orbital_sft/compatibility.yaml}"
run_id="${ORBITAL_RUN_ID:-compatibility-pilot}"
srun --ntasks=1 scripts/orbital_sft pilot --config "$config" --run-id "$run_id"
