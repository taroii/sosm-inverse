#!/usr/bin/env bash
#
# The three cells of Table 1 that did not converge: were they the problem, or
# were we?
#
# WHAT IS ALREADY ESTABLISHED, from the recorded per-evaluation trajectories of
# 409 direct runs. L-BFGS-B's first step, taken with an identity initial
# Hessian against a gradient of order 1e4, overshoots to the LOWER BOUND in
# essentially every run: evaluation 1 is at D_init = 1.2 and evaluation 2 is at
# D = 0.5000 exactly, at every noise level, after which the method recovers in
# one iteration. The bound is --D-min, default 0.5.
#
# The forward solver's documented solvability edge is D ~ 0.48. So every
# inversion in this project, whatever its noise level or true value, routes its
# second forward solve through a point four percent above the edge, where
# Newton needs roughly eight iterations instead of three. That is where the
# ConvergenceError in the logs comes from, and it is a property of the search
# bounds rather than of the data.
#
# WHAT IS NOT ESTABLISHED. Whether that is what killed these three cells. The
# sweep log interleaves concurrent runs, so a traceback in it cannot be
# attributed to a run. Hence this script rather than more log reading.
#
# PHASE A re-runs the three cells unchanged, on current HEAD. If they now
#   succeed, the failures were in code since fixed and the table's denominators
#   change. If they fail the same way, we have a clean reproduction to work on.
# PHASE B re-runs them with the lower bound moved off the cliff to 0.55, which
#   is still far below anything these cells need (the optimum is near 1.0) and
#   still below the smallest true value used anywhere in the paper, 0.6.
#   If Phase A fails and Phase B succeeds, the bound is the cause.
#
# Nothing here is a table number. These runs are a diagnostic, and a run from
# Phase B is NOT comparable with the recorded sweep, which used D_min = 0.5.
#
# Usage:
#     source scripts/env.sh
#     bash scripts/retry_failures.sh
#
# About five minutes, sequential.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="retry-failures.log"

if ! python -c "import firedrake" >/dev/null 2>&1; then
    echo "ERROR: Firedrake not importable. Run: source scripts/env.sh" >&2
    exit 1
fi

run() {
    local label="$1"; shift
    local t0; t0=$(date +%s)
    { echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  $label"; echo "    $*"; } >>"$LOG"
    if "$@" >>"$LOG" 2>&1; then
        printf '  ok    %-44s %4ds\n' "$label" "$(( $(date +%s) - t0 ))"
    else
        printf '  FAIL  %-44s %4ds\n' "$label" "$(( $(date +%s) - t0 ))"
    fi
}

{ echo; echo "############ $(date -u +%Y-%m-%dT%H:%M:%SZ)  retry_failures  \
sha=$(git rev-parse --short HEAD) ############"; } >>"$LOG"
echo "retry_failures starting $(date -u +%H:%M:%SZ)  sha=$(git rev-parse --short HEAD)"
echo "log: $LOG"

echo
echo "PHASE A  the three cells, unchanged, on current HEAD"
run "sigma=1e-3 seed=4" python src/invert.py --seed 4 --sigma 1e-3 \
    --k 4 --N 16 --verbose
for s in 0 4; do
    run "sigma=3e-3 seed=$s" python src/invert.py --seed "$s" --sigma 3e-3 \
        --k 4 --N 16 --verbose
done

echo
echo "PHASE B  the same three with the lower bound off the cliff (D_min = 0.55)"
run "sigma=1e-3 seed=4 D_min=0.55" python src/invert.py --seed 4 --sigma 1e-3 \
    --k 4 --N 16 --D-min 0.55 --verbose
for s in 0 4; do
    run "sigma=3e-3 seed=$s D_min=0.55" python src/invert.py --seed "$s" \
        --sigma 3e-3 --k 4 --N 16 --D-min 0.55 --verbose
done

echo
echo "finished $(date -u +%H:%M:%SZ)"
echo "grep -nE 'D_recovered|ConvergenceError|falling back|search bounds' $LOG"
echo
echo "To bring it back, from the Mac:"
echo "  ssh bruinml 'cd taro/sosm-inverse && tar czf - runs/index.csv \\"
echo "      runs/*/metrics.csv retry-failures.log' | tar xzf -"
