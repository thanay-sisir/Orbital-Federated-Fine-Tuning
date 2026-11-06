#!/usr/bin/env bash
#SBATCH --job-name=orbital-feasibility
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
export CUDA_VISIBLE_DEVICES=""
export ORBITAL_ALLOW_TRAINING=0
: "${ORBITAL_FEASIBILITY_OUTPUT:?Set a new shared output directory}"
policies=(connected fixed contact_aware feasible_fair)
srun --ntasks=1 .venv-orbital/bin/python scripts/orbital_feasibility_sweep.py --policy "${policies[$SLURM_ARRAY_TASK_ID]}" --output "$ORBITAL_FEASIBILITY_OUTPUT"
