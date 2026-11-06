#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
: "${ORBITAL_PYTHON:?Set ORBITAL_PYTHON to an available Python 3.12 interpreter after loading its module}"
"$ORBITAL_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
"$ORBITAL_PYTHON" -m venv .venv-orbital
.venv-orbital/bin/python -m pip install --no-cache-dir --only-binary=:all: -r infra/rivanna/environment/requirements-gpu.txt
.venv-orbital/bin/python -m pip check
.venv-orbital/bin/python -m pip freeze --all > infra/rivanna/environment/resolved.freeze.txt
.venv-orbital/bin/python -m pip inspect > infra/rivanna/environment/resolved.inspect.json
printf '%s\n' 'Environment installed; CUDA/backward compatibility remains unverified until the pilot.'
