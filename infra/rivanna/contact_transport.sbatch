#!/usr/bin/env bash
#SBATCH --job-name=orbital-contact-cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --array=0-9%3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
export CUDA_VISIBLE_DEVICES=""
export ORBITAL_ALLOW_TRAINING=0
contact_output="results/orbital/contact-transport-$SLURM_ARRAY_JOB_ID/case-$SLURM_ARRAY_TASK_ID"
srun --ntasks=1 .venv-orbital/bin/python scripts/orbital_contact_transport.py --case "$SLURM_ARRAY_TASK_ID" --output "$contact_output"
