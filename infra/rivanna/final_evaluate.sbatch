#!/usr/bin/env bash
#SBATCH --job-name=orbital-final-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpupod
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
: "${ORBITAL_CONTROL_RUN:?Set the completed connected epoch two run}"
srun --ntasks=1 .venv-orbital/bin/python scripts/orbital_final_evaluate.py --config configs/orbital_sft/five_gpu_one_epoch.yaml --run-id "final-evaluation-${SLURM_JOB_ID}" --selected results/orbital/epoch-22406221 --control "$ORBITAL_CONTROL_RUN"
