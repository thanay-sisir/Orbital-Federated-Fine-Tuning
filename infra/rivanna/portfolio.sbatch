#!/usr/bin/env bash
#SBATCH --job-name=orbital-portfolio
#SBATCH --account=YOUR_ACCOUNT
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
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
: "${ORBITAL_ENGINEERING_JOB:?Provide the completed engineering-check job ID}"
srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_portfolio.py --engineering-job "$ORBITAL_ENGINEERING_JOB" --recovery "results/orbital/engineering-v1/recovery-$ORBITAL_ENGINEERING_JOB"
srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_cpu_demo.py --output "results/orbital/cpu-demo-$SLURM_JOB_ID"
srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_verify_bundle.py --require-originals
