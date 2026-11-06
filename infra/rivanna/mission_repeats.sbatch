#!/usr/bin/env bash
#SBATCH --job-name=orbital-seed-repeats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:5
#SBATCH --constraint=gpupod
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
policies=(connected fixed feasible_fair)
seed=$((42 + SLURM_ARRAY_TASK_ID / 3))
policy="${policies[$((SLURM_ARRAY_TASK_ID % 3))]}"
srun --ntasks=1 scripts/orbital_sft mission-run --config "configs/orbital_sft/mission_repeats/seed${seed}_${policy}.yaml" --run-id "mission-repeat-v1-s${seed}-${policy}"
