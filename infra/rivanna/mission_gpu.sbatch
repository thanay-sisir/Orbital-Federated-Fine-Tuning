#!/usr/bin/env bash
#SBATCH --job-name=orbital-mission-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:5
#SBATCH --constraint=gpupod
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
policies=(connected fixed feasible_fair)
policy="${policies[$SLURM_ARRAY_TASK_ID]}"
srun --ntasks=1 scripts/orbital_sft mission-run --config "configs/orbital_sft/mission_case62/${policy}.yaml" --run-id "mission-case62-v1-${policy}"
