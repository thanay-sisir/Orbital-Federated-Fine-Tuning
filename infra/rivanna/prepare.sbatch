#!/usr/bin/env bash
#SBATCH --job-name=orbital-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=standard
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
config="${ORBITAL_CONFIG:-configs/orbital_sft/compatibility.yaml}"
export CUDA_VISIBLE_DEVICES=""
srun --ntasks=1 scripts/orbital_sft prepare-data --config "$config"
