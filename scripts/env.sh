#!/usr/bin/env bash
# Common environment for every run on the server. Source this, don't execute it:
#     source scripts/env.sh
# See notes/server.md for the reasoning behind each line.

CONDA_ENV="${CONDA_ENV:-sosm-inverse}"

# Both of Aaron's READMEs require this -- the memory-heavy runs crash without it.
ulimit -s unlimited

# Headless server: never let matplotlib look for a display.
export MPLBACKEND=Agg

# Keep BLAS from oversubscribing when we run several solves concurrently.
# Cores are allocated deliberately per sweep, not grabbed by every library.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "env: conda=$CONDA_DEFAULT_ENV  repo=$REPO_ROOT  stack=$(ulimit -s)"
