#!/usr/bin/env bash
#SBATCH --job-name=orbital-release-assets
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
export CUDA_VISIBLE_DEVICES=""
export ORBITAL_ALLOW_TRAINING=0
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_release_figures.py
