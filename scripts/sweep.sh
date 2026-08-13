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
LOG=sweep-$AXIS.log

run_one() {
    python src/invert.py "$@" >>"$LOG" 2>&1 \
        && echo "ok   $*" || echo "FAIL $*"
}
export -f run_one
export LOG

case "$AXIS" in
    noise)
        # Recovery error against noise level. Includes the baseline cell.
        for s in 1e-4 3e-4 1e-3 3e-3 1e-2; do
            for seed in $SEEDS; do
                echo --sigma "$s" --seed "$seed" --k "$K" --N "$N" --d "$D"
            done
        done
        ;;
    mesh)
        # Recovery error against inversion mesh, data mesh held fixed.
        for n in 8 16 32 64; do
            for seed in $SEEDS; do
                echo --N "$n" --k "$K" --seed "$seed" --sigma 1e-3 --d "$D"
            done
        done
        ;;
    basin)
        # Starting guesses over four orders of magnitude around D_true = 1.
        for init in 0.01 0.1 0.3 3.0 10.0 100.0; do
            for seed in $SEEDS; do
                echo --D-init "$init" --seed "$seed" --k "$K" --N "$N" \
                     --sigma 1e-3 --d "$D"
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
