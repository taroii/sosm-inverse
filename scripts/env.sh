#!/usr/bin/env bash
# Common environment for every run on the server. Source this, don't execute it:
#     source scripts/env.sh
# See notes/server.md for the reasoning behind each line.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# -- Activate Firedrake. -----------------------------------------------------
# Firedrake's installer is apt + pip end to end and never consumes a conda
# package, so a venv is the supported path. Conda is honoured if you set
# CONDA_ENV explicitly, but venv wins when both are present.
VENV="${FIREDRAKE_VENV:-$REPO_ROOT/venv-firedrake}"

if [[ -f "$VENV/bin/activate" ]]; then
    # A conda env active underneath the venv can leave conda's lib/ ahead of the
    # system MPI and HDF5 that PETSc was linked against. Warn, don't deactivate:
    # this shell may belong to another project.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo "WARNING: conda env '${CONDA_DEFAULT_ENV:-?}' is active under the venv." >&2
        echo "         Run 'conda deactivate' before building or running solves." >&2
    fi
    # shellcheck disable=SC1091
    [[ "${VIRTUAL_ENV:-}" == "$VENV" ]] || source "$VENV/bin/activate"
    ACTIVE="venv:$(basename "$VENV")"
elif [[ -n "${CONDA_ENV:-}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
    ACTIVE="conda:$CONDA_ENV"
else
    echo "ERROR: no Firedrake environment found." >&2
    echo "  expected a venv at $VENV" >&2
    echo "  set FIREDRAKE_VENV=/path/to/venv, or CONDA_ENV=name for a conda install" >&2
    return 1 2>/dev/null || exit 1
fi

# PETSC_DIR / PETSC_ARCH / HDF5_MPI, as set at install time by
#     export $(python3 firedrake-configure --show-env)
# Persisted here so every shell gets them without re-running the configure script.
[[ -f "$REPO_ROOT/.firedrake-env" ]] && set -a && source "$REPO_ROOT/.firedrake-env" && set +a

# -- Run discipline. ---------------------------------------------------------

# Both of Aaron's READMEs require this -- the memory-heavy runs crash without it.
ulimit -s unlimited

# Headless server: never let matplotlib look for a display.
export MPLBACKEND=Agg

# Keep BLAS from oversubscribing when we run several solves concurrently.
# Cores are allocated deliberately per sweep, not grabbed by every library.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "env: $ACTIVE  repo=$REPO_ROOT  stack=$(ulimit -s)  petsc=${PETSC_ARCH:-unset}"
