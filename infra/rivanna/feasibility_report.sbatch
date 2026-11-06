#!/usr/bin/env bash
#SBATCH --job-name=orbital-feasibility-report
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
export CUDA_VISIBLE_DEVICES=""
export ORBITAL_ALLOW_TRAINING=0
: "${ORBITAL_FEASIBILITY_OUTPUT:?Set the completed sweep directory}"
: "${ORBITAL_COMPRESSION_OUTPUT:?Set the completed codec probe directory}"
srun --ntasks=1 .venv-orbital/bin/python scripts/orbital_feasibility_report.py --sweep "$ORBITAL_FEASIBILITY_OUTPUT" --compression "$ORBITAL_COMPRESSION_OUTPUT"
