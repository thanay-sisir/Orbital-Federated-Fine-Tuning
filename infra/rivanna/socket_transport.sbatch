#!/usr/bin/env bash
#SBATCH --job-name=orbital-socket-cpu
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
socket_output="results/orbital/socket-transport-$SLURM_JOB_ID"
srun --ntasks=1 .venv-orbital/bin/python -m pytest -c tests/orbital_sft/pytest.ini tests/orbital_sft/test_socket_transport.py tests/orbital_sft/test_reliable_transfer.py --basetemp="$TMPDIR/socket-tests" --junitxml="$accounting/socket-tests.xml"
socket_status=0
srun --ntasks=1 .venv-orbital/bin/python scripts/orbital_socket_demo.py --config configs/orbital_sft/socket_transport.json --output "$socket_output" || socket_status=$?
if [[ -f "$socket_output/results.json" ]]; then
    srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_socket_plots.py --input "$socket_output"
fi
exit "$socket_status"
