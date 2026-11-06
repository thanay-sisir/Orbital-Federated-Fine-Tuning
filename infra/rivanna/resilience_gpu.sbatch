#!/usr/bin/env bash
#SBATCH --job-name=orbital-resilience-gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:5
#SBATCH --constraint=gpupod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
prefix="${ORBITAL_RECOVERY_PREFIX:-resilience-gpu-v1}"
cases=(baseline interrupted)
case_name="${cases[$SLURM_ARRAY_TASK_ID]}"
run_id="$prefix-$case_name"
fault_args=()
if [[ "$case_name" == interrupted ]]; then
    fault_args=(--fault-plan configs/orbital_sft/recovery_v1_faults.json)
fi
for attempt in 0 1 2; do
    resume_args=()
    if (( attempt > 0 )); then
        resume_args=(--resume)
    fi
    set +e
    srun --ntasks=1 --kill-on-bad-exit=1 scripts/orbital_sft mission-run --recoverable --config configs/orbital_sft/recovery_v1.yaml --run-id "$run_id" "${fault_args[@]}" "${resume_args[@]}"
    status=$?
    set -e
    printf '{"attempt":%d,"exit_code":%d}\n' "$attempt" "$status" >> "results/orbital/$run_id/process_outcomes.jsonl"
    if (( status == 0 )); then
        exit 0
    fi
    if (( status != 75 )); then
        exit "$status"
    fi
done
exit 2
