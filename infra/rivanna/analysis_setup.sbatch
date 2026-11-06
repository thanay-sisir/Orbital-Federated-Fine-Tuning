#!/usr/bin/env bash
#SBATCH --job-name=orbital-analysis-setup
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
export PIP_CACHE_DIR="$PWD/.cache/pip-analysis"
.venv-orbital/bin/python -m venv .venv-orbital-analysis
.venv-orbital-analysis/bin/python -m pip install -r infra/rivanna/environment/requirements-analysis.lock.txt
.venv-orbital-analysis/bin/python -m pip check
.venv-orbital-analysis/bin/python -m pip freeze --all > "$accounting/analysis-environment.txt"
