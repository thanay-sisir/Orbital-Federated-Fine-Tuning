#!/usr/bin/env bash
#SBATCH --job-name=orbital-outage-tests
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
srun --ntasks=1 .venv-orbital/bin/python -m pytest -c tests/orbital_sft/pytest.ini tests/orbital_sft --basetemp="$TMPDIR/pytest" --junitxml="$accounting/tests.xml"
