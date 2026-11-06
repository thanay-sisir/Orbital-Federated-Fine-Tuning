#!/usr/bin/env bash
#SBATCH --job-name=orbital-outage-gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:5
#SBATCH --constraint=gpupod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
config=$( .venv-orbital/bin/python -c 'import json,os; print(json.load(open("configs/orbital_sft/outages_selection.json"))["gpu_runs"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["config"])' )
trace=$( .venv-orbital/bin/python -c 'import json,os; print(json.load(open("configs/orbital_sft/outages_selection.json"))["gpu_runs"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["trace"])' )
run_id=$( .venv-orbital/bin/python -c 'import json,os; print(json.load(open("configs/orbital_sft/outages_selection.json"))["gpu_runs"][int(os.environ["SLURM_ARRAY_TASK_ID"])]["run_id"])' )
srun --ntasks=1 scripts/orbital_sft mission-run --recoverable --outage-trace "$trace" --config "$config" --run-id "$run_id"
