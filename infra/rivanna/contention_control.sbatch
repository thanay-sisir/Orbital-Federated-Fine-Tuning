#!/usr/bin/env bash
#SBATCH --job-name=orbital-contention-cpu
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
control_output="results/orbital/contention-control-$SLURM_JOB_ID"
srun --ntasks=1 .venv-orbital/bin/python -m pytest -c tests/orbital_sft/pytest.ini tests/orbital_sft/test_contention_validation.py tests/orbital_sft/test_contact_transport.py tests/orbital_sft/test_socket_transport.py tests/orbital_sft/test_reliable_transfer.py --basetemp="$TMPDIR/contention-tests" --junitxml="$accounting/contention-tests.xml"
control_status=0
srun --ntasks=1 .venv-orbital/bin/python scripts/orbital_contention_control.py --output "$control_output" || control_status=$?
if [[ -f "$control_output/payload_validation.json" ]]; then
    srun --ntasks=1 .venv-orbital-analysis/bin/python scripts/orbital_contention_plots.py --input "$control_output"
    cp "$control_output/payload_validation.json" docs/orbital_sft/evidence/contention_control_results.json
    cp "$control_output/payload_report.md" docs/orbital_sft/evidence/contention_control_report.md
fi
exit "$control_status"
