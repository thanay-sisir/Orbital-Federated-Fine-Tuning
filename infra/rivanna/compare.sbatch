#!/usr/bin/env bash
#SBATCH --job-name=orbital-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpupod
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
config="${ORBITAL_CONFIG:-configs/orbital_sft/five_gpu_one_epoch.yaml}"
candidate="${ORBITAL_CANDIDATE:-results/orbital/reference-22367142}"
arguments=(--config "$config" --run-id "comparison-${SLURM_JOB_ID}" --candidate "$candidate")
arguments+=(--benchmark-dir "${ORBITAL_BENCHMARK_DIR:-artifacts/benchmarks/development-v1}")
for extra in "${ORBITAL_CANDIDATE2:-}" "${ORBITAL_CANDIDATE3:-}"; do
    if [[ -n "$extra" ]]; then
        arguments+=(--candidate "$extra")
    fi
done
if [[ "${ORBITAL_FINAL_TEST:-0}" == "1" ]]; then
    arguments+=(--final-test)
fi
srun --ntasks=1 .venv-orbital/bin/python scripts/orbital_compare.py "${arguments[@]}"
