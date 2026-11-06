#!/usr/bin/env bash
#SBATCH --job-name=orbital-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpupod
#SBATCH --array=0-5%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
configs=(seed42-fixed seed42-aware seed43-fixed seed43-aware seed44-fixed seed44-aware)
config="configs/orbital_sft/generated/${configs[$SLURM_ARRAY_TASK_ID]}.yaml"
test -f "$config"
srun --ntasks=1 scripts/orbital_sft run --config "$config" --run-id "sweep-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
