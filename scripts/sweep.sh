#!/usr/bin/env bash
#
# Drive the inversion grid. One process per run, several at a time.
#
# Concurrency is bounded by MEMORY, not cores. From bench.csv, one inversion at
# k=4 holds roughly 0.6 GB at N=16, 1.6 GB at N=32 and 5.8 GB at N=64, against
# 48 GB total. JOBS defaults to 6, which is safe up to N=64; raise it for the
# smaller meshes.
#
# Every run writes its own directory, so runs are independent and the sweep can
# be interrupted and restarted. runlog skips nothing, though -- re-running
# repeats work, so narrow the axes rather than re-running the lot.
#
# Usage:
#     tmux new -s sweep
#     source scripts/env.sh
#     bash scripts/sweep.sh noise
#     JOBS=12 N=16 bash scripts/sweep.sh basin

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

AXIS="${1:-noise}"
JOBS="${JOBS:-6}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
K="${K:-4}"
N="${N:-16}"
D="${D:-2}"
D_INIT="${D_INIT:-1.2}"
# The paper's prototype puts the prior ABOVE the truth (1.4286x) and the initial
# guess further above (3.4286x), and attributes its residual bias to the prior's
# pull. We keep the prior ratio; D_INIT stays at 1.2 for the noise and mesh axes
# because the basin axis is where the initial guess is the variable under study.
# Set D_INIT=3.4286 to mirror the prototype exactly -- verify one run first, it
# needs four continuation steps per objective evaluation rather than one.
D_PRIOR="${D_PRIOR:-1.4286}"
LOG=sweep-$AXIS.log

run_one() {
    python src/invert.py "$@" >>"$LOG" 2>&1 \
        && echo "ok   $*" || echo "FAIL $*"
}

# Warm the clean-data cache serially before fanning out. The fine-mesh solve is
# identical across seeds, but if N jobs start together they all miss the cache
# and each builds an 11 GB solve at once. One cheap serial run first, then the
# rest hit the cache and hold only their own inversion.
warm_cache() {
    echo "warming data cache (one fine-mesh solve)..."
    python src/invert.py "$@" --check-gradient >>"$LOG" 2>&1 \
        && echo "cache warm" || echo "WARN: cache warm failed, see $LOG"
}
export -f run_one
export LOG

# The log is appended, so mark where each sweep begins. Without this, grepping
# for failures mixes this run with every previous one -- an earlier diagnosis
# counted 60 solver failures that mostly belonged to a superseded sweep.
{ echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  axis=$AXIS  K=$K N=$N D=$D \
D_INIT=$D_INIT D_PRIOR=$D_PRIOR  sha=$(git rev-parse --short HEAD) ==="; } >>"$LOG"

warm_cache --k "$K" --N "$N" --d "$D" --sigma 1e-3 --seed 0 \
           --D-init "$D_INIT" --D-prior "$D_PRIOR"

case "$AXIS" in
    noise)
        # Recovery error against noise level. Includes the baseline cell.
        for s in 1e-4 3e-4 1e-3 3e-3 1e-2; do
            for seed in $SEEDS; do
                echo --sigma "$s" --seed "$seed" --k "$K" --N "$N" --d "$D" \
                     --D-init "$D_INIT" --D-prior "$D_PRIOR"
            done
        done
        ;;
    mesh)
        # Recovery error against inversion mesh, data mesh held fixed.
        for n in 8 16 32 64; do
            for seed in $SEEDS; do
                echo --N "$n" --k "$K" --seed "$seed" --sigma 1e-3 --d "$D" \
                     --D-init "$D_INIT" --D-prior "$D_PRIOR"
            done
        done
        ;;
    basin)
        # Starting guesses spanning the SOLVABLE range, not four orders of
        # magnitude. Continuation fails below D ~ 0.45 for this configuration
        # (E11, and the trace in E10b's neighbourhood), so 0.01 and 0.1 are not
        # hard starting guesses -- they are guesses with no forward solution, and
        # every such cell would have failed rather than reported a wide basin.
        # The upper edge is not yet measured; 8.0 sits inside the default bound.
        for init in 0.6 0.8 1.5 2.5 4.0 8.0; do
            for seed in $SEEDS; do
                echo --D-init "$init" --seed "$seed" --k "$K" --N "$N" \
                     --sigma 1e-3 --d "$D" --D-prior "$D_PRIOR"
            done
        done
        ;;
    *)
        echo "unknown axis: $AXIS  (noise|mesh|basin)" >&2
        exit 2
        ;;
esac | xargs -P "$JOBS" -I{} bash -c 'run_one {}'

echo
echo "log: $LOG"
echo "results: runs/index.csv and runs/*/metrics.csv"
