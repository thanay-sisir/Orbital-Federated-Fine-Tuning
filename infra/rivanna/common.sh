#!/usr/bin/env bash
set -euo pipefail
cd "${ORBITAL_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${SLURM_JOB_ID:?Run this script through sbatch}"
export PYTHONPATH="$PWD/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export ORBITAL_ALLOW_TRAINING=1
unset OPENROUTER_API_KEY OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p logs "results/orbital-accounting/$SLURM_JOB_ID" "$PWD/.cache/tmp/$SLURM_JOB_ID"
export TMPDIR="$PWD/.cache/tmp/$SLURM_JOB_ID"
accounting="$PWD/results/orbital-accounting/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$accounting/job-start.txt"
module list > "$accounting/modules.txt" 2>&1 || true
.venv-orbital/bin/python -m pip freeze --all > "$accounting/environment.txt"
sha256sum infra/rivanna/environment/requirements-gpu.txt > "$accounting/requirements.sha256"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$accounting/exit-code.txt"
    sacct -j "$SLURM_JOB_ID" --parsable2 --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS,TotalCPU > "$accounting/sacct.psv" 2>&1 || true
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
