#!/usr/bin/env bash
#SBATCH --job-name=orbital-outage-geometry
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
export CUDA_VISIBLE_DEVICES=""
export ORBITAL_ALLOW_TRAINING=0
srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_validate_geometry.py --run "results/orbital/outages-v1/cpu/geometry-$SLURM_ARRAY_TASK_ID" --output "results/orbital/outages-v1/geometry-check-$SLURM_ARRAY_TASK_ID.json"
