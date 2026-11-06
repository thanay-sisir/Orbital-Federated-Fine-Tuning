#!/usr/bin/env bash
#SBATCH --job-name=orbital-contact-report
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${ORBITAL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}/infra/rivanna/common.sh"
export CUDA_VISIBLE_DEVICES=""
export ORBITAL_ALLOW_TRAINING=0
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
: "${ORBITAL_CONTACT_ARRAY:?Supply the CPU array job ID}"
: "${ORBITAL_CONTACT_TESTS:?Supply the CPU tests job ID}"
srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_contact_report.py --array-job "$ORBITAL_CONTACT_ARRAY" --tests-job "$ORBITAL_CONTACT_TESTS" --output "results/orbital/contact-transport-$ORBITAL_CONTACT_ARRAY"
